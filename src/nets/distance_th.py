import numpy as np
import torch

class LogDet(torch.autograd.Function):
    """
    Matrix log determinant. Input should be a square matrix
    """

    @staticmethod
    def forward(ctx, x, eps=0.0):
        output = torch.log(x.potrf().diag() + eps).sum() * 2
        output = x.new([output])
        ctx.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_variables
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * x.inverse().t()
        return grad_input


def logdet(x):
    return LogDet.apply(x)


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


def gauss_kld(xs, ys, use_logdet=False, eps=1e-6): # float(np.finfo(np.float32).eps)):
    n_batch, n_hidden = xs.size()
    xm = xs.mean(0, keepdim=True)
    ym = ys.mean(0, keepdim=True)
    xcov = cov(xs, xm)
    ycov = cov(ys, ym)
    xcov += torch.diag(xcov.diag() + eps)
    ycov += torch.diag(ycov.diag() + eps)
    log_ratio = logdet(ycov) - logdet(xcov)
    ycovi = ycov.inverse()
    xym = xm - ym  # (1, n_hidden)
    hess = xym.mm(ycovi).mm(xym.t())
    tr = torch.trace(ycovi.mm(xcov))
    return 0.5 * (log_ratio + tr + hess - n_hidden).squeeze()


def packed_gauss_kld(hspad, hslens, htpad, htlens):
    from torch.nn.utils.rnn import pack_padded_sequence
    hspack = pack_padded_sequence(hspad, hslens, batch_first=True)
    htpack = pack_padded_sequence(htpad, htlens, batch_first=True)
    return gauss_kld(hspack.data, htpack.data)


def mmd_loss(xs,ys,beta=1.0):
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


def packed_mmd(hspad, hslens, htpad, htlens):
    from torch.nn.utils.rnn import pack_padded_sequence
    hspack = pack_padded_sequence(hspad, hslens, batch_first=True)
    htpack = pack_padded_sequence(htpad, htlens, batch_first=True)
    return mmd_loss(hspack.data, htpack.data)
