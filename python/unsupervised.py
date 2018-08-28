import logging

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import six

import e2e_asr_attctc_th as base


def mmd(xs,ys,beta=1.0):
    Nx = xs.shape[0]
    Ny = ys.shape[0]
    Kxy = torch.matmul(xs,ys.t())
    dia1 = torch.sum(xs*xs,1)
    dia2 = torch.sum(ys*ys,1)
    Kxy = Kxy-0.5*dia1.unsqueeze(1).expand(Nx,Ny)
    Kxy = Kxy-0.5*dia2.expand(Nx,Ny)
    Kxy = torch.exp(beta*Kxy).sum()/Nx/Ny

    Kx = torch.matmul(xs,xs.t())
    Kx = Kx-0.5*dia1.unsqueeze(1).expand(Nx,Nx)
    Kx = Kx-0.5*dia1.expand(Nx,Nx)
    Kx = torch.exp(beta*Kx).sum()/Nx/Nx

    Ky = torch.matmul(ys,ys.t())
    Ky = Ky-0.5*dia2.unsqueeze(1).expand(Ny,Ny)
    Ky = Ky-0.5*dia2.expand(Ny,Ny)
    Ky = torch.exp(beta*Ky).sum()/Ny/Ny

    return Kx+Ky-2*Kxy


class _Det(torch.autograd.Function):
    """
    Matrix determinant. Input should be a square matrix
    """

    @staticmethod
    def forward(ctx, x):
        output = x.potrf().diag().prod()**2
        output = x.new([output])
        ctx.save_for_backward(x, output)
        # ctx.save_for_backward(u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_variables
        # u, output = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            # TODO TEST
            grad_input = grad_output * output * x.inverse().t()
            # grad_input = grad_output * output * torch.potrf(u).t()

        return grad_input

def det(x):
    # u = torch.potrf(x)
    return _Det.apply(x)


class LogDet(torch.autograd.Function):
    """
    Matrix log determinant. Input should be a square matrix
    """

    @staticmethod
    def forward(ctx, x, eps=0.0):
        output = torch.log(x.potrf().diag() + eps).sum() * 2
        output = x.new([output])
        ctx.save_for_backward(x, output)
        # ctx.save_for_backward(u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_variables
        # u, output = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            # TODO TEST
            grad_input = grad_output * x.inverse().t()
            # grad_input = grad_output * torch.potrf(u).t()
        return grad_input

def logdet(x):
    # u = torch.potrf(x)
    return LogDet.apply(x)


def test_det():
    x = Variable(torch.rand(3, 3) / 10.0 + torch.eye(3).float(), requires_grad=True)
    torch.autograd.gradcheck(det, (x,), eps=1e-4, atol=0.1, rtol=0.1)

def test_logdet():
    x = Variable(torch.rand(3, 3) + torch.eye(3).float() * 3 , requires_grad=True)
    d = det(x).log()
    d.backward()
    gd = x.grad.clone()
    ld = logdet(x)
    x.grad = None
    ld.backward()
    gld = x.grad
    np.testing.assert_allclose(d.data.numpy(), ld.data.numpy())
    np.testing.assert_allclose(gd.data.numpy(), gld.data.numpy())

def cov(xs, m=None):
    assert xs.dim() == 2
    if m is None:
        m = xs.mean(0, keepdim=True)
    assert m.size() == (1, xs.size(1))
    return (xs - m).t().mm(xs - m) / xs.size(0)

threshold = torch.nn.functional.threshold

def unclamp_(x, eps):
    """
    >>> a = torch.FloatTensor([0.0, 1.0, -0.1, 0.1])
    >>> unclamp(a, 0.5)
    [0.5, 1.0, -0.5, 0.5]
    """
    ng = x.abs() < eps
    sign = x.sign()
    fill_value = sign.float() * eps + (sign == 0).float() * eps
    return x.masked_fill_(ng, 0) + ng.float() * fill_value

def gauss_kld(xs, ys, use_logdet=False, eps=float(np.finfo(np.float32).eps)):
    n_batch, n_hidden = xs.size()
    xm = xs.mean(0, keepdim=True)
    ym = ys.mean(0, keepdim=True)
    xcov = cov(xs, xm)
    ycov = cov(ys, ym)
    xcov += torch.diag(xcov.diag() + eps)
    ycov += torch.diag(ycov.diag() + eps)
    if use_logdet:
        log_ratio = logdet(ycov) - logdet(xcov)
    else:
        log_ratio = torch.log(threshold(det(ycov), eps, eps)) - torch.log(threshold(det(xcov), eps, eps))
    ycovi = ycov.inverse()
    xym = xm - ym  # (1, n_hidden)
    hess = xym.mm(ycovi).mm(xym.t())
    tr = torch.trace(ycovi.mm(xcov))
    return 0.5 * (log_ratio + tr + hess - n_hidden).squeeze()


class EmbedRNN(torch.nn.Module):
    def __init__(self, n_in, n_out, n_layers=1):
        super(EmbedRNN, self).__init__()
        self.embed = torch.nn.Embedding(n_in, n_out)
        self.rnn = torch.nn.LSTM(n_out, n_out, n_layers,
                                 bidirectional=True, batch_first=True)
        self.merge = torch.nn.Linear(n_out * 2, n_out)

    def forward(self, xpad, xlen):
        """
        :param xpad: (batchsize x max(xlen)) LongTensor
        :return hpad: (batchsize x max(xlen) x n_out) FloatTensor
        :return hlen: length list of int. hlen == xlen
        """
        h = self.embed(xpad)
        hpack = pack_padded_sequence(h, xlen, batch_first=True)
        hpack, states = self.rnn(hpack)
        hpad, hlen = pad_packed_sequence(hpack, batch_first=True)
        b, t, o = hpad.shape
        hpad = self.merge(hpad.contiguous().view(b * t, o)).view(b, t, -1)
        return hpad, hlen


class MMSEDecoder(torch.nn.Module):
    """
    hidden-to-speech decoder with a MMSE criterion

    TODO(karita): use Tacotron-like structure
    """
    def __init__(self, eprojs, odim, dlayers, dunits, att, verbose=0):
        super(MMSEDecoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.in_linear = torch.nn.Linear(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.verbose = verbose

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ypad, ylen):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = base.mask_by_length(hpad, hlen, 0)
        self.loss = None

        # get dim, length info
        batch = ypad.size(0)
        olength = ypad.size(1)

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        att_weight_all = []  # for debugging

        # pre-computation of embedding
        eys = self.in_linear(ypad.view(batch * olength, -1)).view(batch, olength, -1)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])
            att_weight_all.append(att_w.data)  # for debugging

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all).view(batch, olength, -1)
        ym = base.mask_by_length(y_all, ylen)
        tm = base.mask_by_length(ypad, ylen)
        self.loss = torch.sum((ym - tm) ** 2)
        self.loss *= (np.mean(ylen))
        logging.info('att loss:' + str(self.loss.data))
        return self.loss, att_weight_all


class Discriminator(torch.nn.Module):
    def __init__(self, idim, odim):
        super(Discriminator, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.ReLU(),
            torch.nn.Linear(odim, odim),
            torch.nn.ReLU(),
            torch.nn.Linear(odim, 1)
        )

    def forward(self, spack, tpack):
        ns = spack.size(0)
        nt = tpack.size(0)
        input = torch.cat((spack, tpack), dim=0)
        predict = self.seq(input)
        target = input.data.new(ns + nt, 1)
        target[:ns] = 0
        target[ns:] = 1
        target = Variable(target)
        return -torch.nn.functional.binary_cross_entropy_with_logits(predict, target)


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir

        if hasattr(args, "unsupervised_loss"):
            self.unsupervised_loss = args.unsupervised_loss
        else:
            self.unsupervised_loss = None
        if hasattr(args, "use_batchnorm") and args.use_batchnorm:
            self.batchnorm = torch.nn.BatchNorm1d(args.eprojs)
        else:
            self.batchnorm = None

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # encoder
        self.enc_t = EmbedRNN(odim, args.eprojs)
        self.enc = base.Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                                  self.subsample, args.dropout_rate)
        self.enc_common_rnn = getattr(self.enc.enc1, "bilstm%d" % (args.elayers-1))
        self.enc_common_merge = getattr(self.enc.enc1, "bt%d" % (args.elayers-1))

        # ctc
        self.ctc = base.CTC(odim, args.eprojs, args.dropout_rate)

        # attention
        if args.atype == 'dot':
            self.att = base.AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location': # 
            self.att = base.AttLoc(args.eprojs, args.dunits,
                                     args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'noatt':
            self.att = base.NoAtt()
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
        # if args.tied_attention:
        #     self.att_s = self.att

        # decoder
        self.dec = base.Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                                  self.sos, self.eos, self.att, self.verbose, self.char_list)
        # self.dec_s = MMSEDecoder(args.eprojs, idim, args.dlayers, args.dunits,
        #                          self.att_s, self.verbose)
        # if args.tied_decoder:
        #     self.dec_s.decoder = self.dec.decoder

        # weight initialization
        self.init_like_chainer()

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        base.lecun_normal_init_parameters(self)

        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        self.enc_t.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            base.set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def sort_variables(self, xs, sorted_index):
        xs = [xs[i] for i in sorted_index]
        xs = [base.to_cuda(self, Variable(torch.from_numpy(xx))) for xx in xs]
        xlens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        return xs, xlens

    def forward_common(self, xpad, xlen):
        # hpad, hlen = self.enc_common_rnn(xpad, xlen)
        xpack = pack_padded_sequence(xpad, xlen, batch_first=True)
        hpack, states = self.enc_common_rnn(xpack)
        hpad, hlen = pad_packed_sequence(hpack, batch_first=True)
        b, t, o = hpad.shape
        hpad = torch.tanh(self.enc_common_merge(hpad.contiguous().view(b * t, o)).view(b, t, -1))
        return hpad, hlen

    def forward(self, data, supervised=False, discriminator=None, only_encoder=False):
        '''E2E forward (unsupervised)

        :param data:
        :return:
        '''
        # utt list of frame x dim
        xs = [d[1]['feat'] for d in data]
        tids = [d[1]['tokenid'].split() for d in data]
        ys = [np.fromiter(map(int, t), dtype=np.int64) for t in tids]

        # sort by length
        sorted_index = sorted(range(len(xs)), key=lambda i: -len(xs[i]))
        xs, xlens = self.sort_variables(xs, sorted_index)
        ys, ylens = self.sort_variables(ys, sorted_index)

        # ys = [base.to_cuda(self, Variable(torch.from_numpy(y))) for y in ys]
        if supervised or not self.training:
            # forward encoder for speech
            xpad = base.pad_list(xs)
            hxpad, hxlens = self.enc(xpad, xlens)
            if self.batchnorm:
                hxpack = pack_padded_sequence(hxpad, hxlens, batch_first=True)
                hxpack = PackedSequence(self.batchnorm(hxpack.data), hxpack.batch_sizes)
                hxpad, hxlens = pad_packed_sequence(hxpack, batch_first=True)

            # CTC loss
            loss_ctc = self.ctc(hxpad, hxlens, ys)

            # forward decoders
            loss_att, acc, att_t = self.dec(hxpad, hxlens, ys)
            return loss_ctc, loss_att, acc

            # loss_speech, att_s = self.dec_s(hxpad, hxlens, xpad, xlens)
        else:
            # forward encoder for text
            y_sorted_index = sorted(range(len(ys)), key=lambda i: -len(ys[i]))
            ys = [ys[i] for i in y_sorted_index]
            ylens = [ylens[i] for i in y_sorted_index]
            ypad = base.pad_list(ys, 0)
            hypad, hylens = self.enc_t(ypad, ylens)

            # forward common encoder
            hypad, hylens = self.forward_common(hypad, hylens)
            hypack = pack_padded_sequence(hypad, hylens, batch_first=True)

            if self.unsupervised_loss is not None and self.unsupervised_loss != "None":
                xpad = base.pad_list(xs)
                hxpad, hxlens = self.enc(xpad, xlens)
                hxpack = pack_padded_sequence(hxpad, hxlens, batch_first=True)
                if self.batchnorm:
                    hxpack = PackedSequence(self.batchnorm(hxpack.data), hxpack.batch_sizes)
                    hypack = PackedSequence(self.batchnorm(hypack.data), hypack.batch_sizes)

                if only_encoder:
                    return hxpack, hypack

                if self.unsupervised_loss == "variance":
                    loss_unsupervised = torch.cat((hxpack.data, hypack.data), dim=0).var(1).mean()
                if self.unsupervised_loss == "gauss":
                    loss_unsupervised = gauss_kld(hxpack.data, hypack.data)
                if self.unsupervised_loss == "gausslogdet":
                    loss_unsupervised = gauss_kld(hxpack.data, hypack.data, use_logdet=True)
                if self.unsupervised_loss == "mmd":
                    loss_unsupervised = mmd(hxpack.data, hypack.data)
                if self.unsupervised_loss == "gan":
                    loss_unsupervised = discriminator(hxpack.data, hypack.data)
            else:
                loss_unsupervised = 0.0
                if only_encoder:
                    xpad = base.pad_list(xs)
                    hxpad, hxlens = self.enc(xpad, xlens)
                    hxpack = pack_padded_sequence(hxpad, hxlens, batch_first=True)
                    return hxpack, hypack

            # 3. forward decoders
            loss_text, acc, att_t = self.dec(hypad, hylens, ys)
            # loss_speech, att_s = self.dec_s(hxpad, hxlens, xpad, xlens)
            return loss_text, loss_unsupervised, acc


    def recognize(self, x, recog_args, char_list):
        '''E2E greedy/beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        xlen = [x.shape[0]]
        xpad = base.to_cuda(self, Variable(torch.from_numpy(
            np.array(x, dtype=np.float32)), volatile=True))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h, hlen = self.enc(xpad.unsqueeze(0), xlen)
        # h, hlen = self.forward_common(h, hlen)
        lpz = None

        # 2. decoder
        # decode the first utterance
        if recog_args.beam_size == 1:
            y = self.dec.recognize(h[0], recog_args)
        else:
            y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list)

        if prev:
            self.train()
        return y

