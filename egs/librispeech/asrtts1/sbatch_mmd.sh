
ae_weight=0.1
etype=vggblstmp
exp_prefix="bs32_lr1e-3_${etype}/sbatch"
for ngpu in 1 ; do
    for lr in 1e-3; do
        for bs in 32; do
            for mmd in 0.1 2.0 10.0; do
                # full
                ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 1.0 --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} &
                sleep 2
                # asr + s2s + t2t
                ./run2.sh --stage 5 --ngpu ${ngpu} --etype ${etype} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 0.0 --s2s_weight ${ae_weight} --t2t_weight ${ae_weight} &
                sleep 2
            done
        done
    done
done
wait
