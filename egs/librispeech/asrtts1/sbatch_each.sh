prefix="each2/sbatch"
for ngpu in 1 ; do
    for lr in 1e-4; do
        for bs in 32; do
            for mmd in 0.0 1.0; do
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 1.0 --s2s_weight 1.0 --t2t_weight 1.0 &
                sleep 2
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 0.0 --s2s_weight 0.0 --t2t_weight 0.0 &
                sleep 2
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 0.0 --tts_weight 1.0 --s2s_weight 0.0 --t2t_weight 0.0 &
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 1.0 --s2s_weight 1.0 --t2t_weight 0.0 &
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --exp_prefix ${exp_prefix} --asr_weight 1.0 --tts_weight 1.0 --s2s_weight 0.0 --t2t_weight 0.1 &
                sleep 2
            done
        done
    done
done
wait
