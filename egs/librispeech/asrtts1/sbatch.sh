for ngpu in 1 ; do
    for lr in 1e-2 1e-3 1e-4; do
        for bs in 32 64 96; do
            for mmd in 1.0 0.5 0.0; do
                ./run.sh --stage 5 --ngpu ${ngpu} --lr ${lr} --batchsize ${bs} --mmd_weight ${mmd} --tag sbatch2_ngpu${ngpu}_lr${lr}_bs${bs}_mmd${mmd} &
                sleep 2
            done
        done
    done
done
wait
