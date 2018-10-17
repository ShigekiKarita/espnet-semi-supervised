asr_weight=0.1
ae_weight=0.01
etype=vggblstmp
tts="/data/work70/skarita/exp/espnet-asrtts/egs/librispeech/asrtts1/exp/train_clean_100_data_short_tts_pre/results/model.loss.best"
asr="/data/work70/skarita/exp/espnet-asrtts/egs/librispeech/asrtts1/exp/train_clean_100_data_short_asr_vggblstmp_32/results/model.acc.best"
opt=" --model_asr ${asr} --model_tts ${tts} --inter_domain_loss kl "
ngpu=1
for tts_weight in 0.01 ; do
    for lr in 1e-5; do
        for bs in 32; do
            exp_prefix="1016_kl_run5_retrain_bs32_lr${lr}_asr${asr_weight}_${etype}/sbatch"
            for mmd in 0.1 0.01 ; do
                # full
                ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight ${tts_weight} --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} ${opt} &
                sleep 2
                # asr + s2s + t2t
                ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight 0.0 --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} ${opt} &
                sleep 2
            done

            # mmd=0.0
            # # asr
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight 0.0 --s2s_weight 0.0 --t2t_weight 0.0 ${opt} &
            # sleep 2

            # # tts
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.0 --tts_weight ${tts_weight} --s2s_weight 0.0 --t2t_weight 0.0 ${opt} &
            # sleep 2

            # # asr + tts
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight ${tts_weight} --s2s_weight 0.0 --t2t_weight 0.0 ${opt} &
            # sleep 2
            # # asr + s2s
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight 0.0 --s2s_weight ${ae_weight} --t2t_weight 0.0 ${opt} &
            # sleep 2

            # # asr + t2t
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight 0.0 --s2s_weight 0.0 --t2t_weight ${ae_weight} ${opt} &
            # sleep 2

            # # asr + tts + s2s
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight ${tts_weight} --s2s_weight ${ae_weight} --t2t_weight 0.0 ${opt} &
            # sleep 2

            # # asr + tts + t2t
            # ./run5.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight ${asr_weight} --tts_weight ${tts_weight} --s2s_weight 0.0 --t2t_weight ${ae_weight} ${opt} &
            # sleep 2
        done
    done
done
wait
