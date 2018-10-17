
ae_weight=0.01
etype=vggblstmp
exp_prefix="1001_bs32_lr1e-3_${etype}/sbatch"
for ngpu in 1 ; do
    for lr in 1e-3; do
        for bs in 32; do
            for mmd in 1.0 0.0; do
                # full
                ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 1.0 --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} &
                sleep 2
                # asr + s2s + t2t
                ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 0.0 --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} &
                sleep 2
            done

            # asr
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 0.0 --s2s_weight 0.0 --t2t_weight 0.0 &
            sleep 2

            # tts
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.0 --tts_weight 1.0 --s2s_weight 0.0 --t2t_weight 0.0 &
            sleep 2

            # asr + tts
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 1.0 --s2s_weight 0.0 --t2t_weight 0.0 &
            sleep 2
            # asr + s2s
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 0.0 --s2s_weight ${ae_weight} --t2t_weight 0.0 &
            sleep 2

            # asr + t2t
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 0.0 --s2s_weight 0.0 --t2t_weight ${ae_weight} &
            sleep 2

            # asr + tts + s2s
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 1.0 --s2s_weight ${ae_weight} --t2t_weight 0.0 &
            sleep 2

            # asr + tts + t2t
            ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.1 --tts_weight 1.0 --s2s_weight 0.0 --t2t_weight ${ae_weight} &
            sleep 2
        done
    done
done
wait
