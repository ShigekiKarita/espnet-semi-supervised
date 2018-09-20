for ngpu in 1 ; do
    for lr in 1e-4 1e-5; do
        for bs in 32; do
            for mmd in 1.0 10.0 0.0; do
                ./run2.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --tag sbatch_equal_weight_ngpu${ngpu}_lr${lr}_bs${bs}_mmd${mmd} --asr_weight 1.0 --tts_weight 1.0 --s2s_weight 1.0 --t2t_weight 1.0 &
                sleep 2
            done
        done
    done
done
wait
