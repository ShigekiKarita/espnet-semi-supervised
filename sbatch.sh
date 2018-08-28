#!/usr/bin/env zsh

base=$(dirname $0)
mkdir -p ./slurm/log
jobid_list=()

n_parallel=1

submit() {
    num_gpu=$1
    logname=$2
    command=${@:3}
    script="./slurm/${logname}.sh"
    echo "#!/usr/bin/env zsh" > $script
    echo "${command}" '|| {echo + dead-jobid: $SLURM_JOB_ID; echo + command:' "$command" "; echo + logfile: \
    $logname; tail $logname } | mattersend -c karita-exp " >> $script
    msg=$(sbatch -p gpu  --gres gpu:$num_gpu -c 1 -N 1 -o $logname -e $logname $script)
    jobid=$(echo $msg | awk '{print $NF}')
    jobid_list+=($jobid)

    echo "${command}"
    echo "${msg}"
}

mkdir -p log model


opt=adadelta
alpha=0.5
batch=30
elayers=6
batchnorm=False
dlayers=1
atype=location
epochs=15
for unsupervised_loss in mmd gausslogdet gan None; do
    for sup_data_ratio in 1.00 0.75 0.50 0.25; do
        for sup_loss_ratio in 0.9 0.5 0.1 ; do
            for wd in 0.0 ; do
                for st_ratio in 0.9 0.5 0.1 ; do
                    for lr in 1.0; do
                        exp_name=sbatch_${unsupervised_loss}_alpha${alpha}_bn${batchnorm}_${opt}_lr${lr}_bs${batch}_el${elayers}_dl${dlayers}_att_${atype}_batch${batch}_data${sup_data_ratio}_loss${sup_loss_ratio}_st${st_ratio}_epochs${epochs}
                        log_name=log/${train_set}_${exp_name}.log
                        ngpu=1

                        echo $exp_name
                        submit $ngpu $log_name OMP_NUM_THREADS=1 ./run.sh \
                               --stage 3 --tag $exp_name \
                               --gpu 0 \
                               --unsupervised_loss ${unsupervised_loss} \
                               --mtlalpha $alpha \
                               --batchsize $batch \
                               --elayers $elayers \
                               --dlayers $dlayers \
                               --opt $opt \
                               --epochs $epochs \
                               --backend pytorch \
                               --etype blstmp \
                               --atype $atype \
                               --weight_decay $wd \
                               --st_ratio $st_ratio \
                               --sup_loss_ratio $sup_loss_ratio \
                               --sup_data_ratio $sup_data_ratio \
                               --use_batchnorm $batchnorm
                        sleep 3
                    done
                done
            done
        done
    done
done

echo "============================="
echo "to delete batch jobs: scancel ${jobid_list[@]}"
echo "============================="

ch=./scancel.sh
echo "#!/bin/sh" > $ch
echo scancel ${jobid_list[@]} >> $ch
chmod +x $ch
