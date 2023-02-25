if [ $# != 4 ] && [ $# != 5 ]
then
    echo "Usage: bash run_train.sh [CONFIG_PATH] [DEVICE_TRAGET] [DEVICE_NUM] [DEVICE_ID|RANK_TABLE_FILE|CUDA_VISIBLE_DEVICES]"
    echo "Usage: bash run_train.sh [CONFIG_PATH] [DEVICE_TRAGET] [DEVICE_NUM] [DEVICE_ID|RANK_TABLE_FILE|CUDA_VISIBLE_DEVICES] [USE_MPI]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

CONFIG_PATH=$(get_real_path $1)
DEVICE_TRAGET=$2
export DEVICE_NUM=$3
export RANK_SIZE=$DEVICE_NUM

PARALLEL=0
if [ $DEVICE_NUM -gt 1 ]; then
    PARALLEL=1
fi

USE_MPI=1
if [ $# == 5 ]; then
    USE_MPI=$5
    if [ $DEVICE_TRAGET == 'GPU' ] && [ $USE_MPI == 0 ] && [ $PARALLEL == 1 ]; then
        USE_MPI=1
        echo "Run parallel train on GPU only support OpenMPI."
    fi
fi

if [ $DEVICE_TRAGET == 'Ascend' ]; then
    echo "Run Ascend"
    if [ $DEVICE_NUM == 1 ]; then
        export DEVICE_NUM=1
        export DEVICE_ID=$4
        export RANK_SIZE=1
        export RANK_ID=$DEVICE_ID
    elif [ $DEVICE_NUM == 8 ]; then
        export DEVICE_NUM=8
        export RANK_SIZE=8
        export RANK_TABLE_FILE=$4
        export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE
        PARALLEL=1
    else
        echo "error: Ascend device num not equal 1 or 8"
        exit 1
    fi
elif [ $DEVICE_TRAGET == 'GPU' ]; then
    echo "Run GPU"
    export CUDA_VISIBLE_DEVICES=$4
elif [ $DEVICE_TRAGET == "CPU" ]; then
    echo "Run CPU"
    if [ $DEVICE_NUM -gt 1 ]; then
        echo "error: CPU device num not equal 1"
        exit 1
    fi
else
    echo "error: Not support $DEVICE_TRAGET platform."
    exit 1
fi


if [ ! -f $CONFIG_PATH ]; then
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
    exit 1
fi


if [ $PARALLEL == 1 ]; then
    if [ $USE_MPI == 1 ]; then
        mpirun --allow-run-as-root -n $RANK_SIZE --merge-stderr-to-stdout \
        python ./tools/train.py \
            --config=$CONFIG_PATH \
            --device_target=$DEVICE_TRAGET \
            --is_parallel=True > log.txt 2>&1 &
    else
        cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
        avg=`expr $cpus \/ $RANK_SIZE`
        gap=`expr $avg \- 1`

        rm -rf ./train_parallel
        mkdir ./train_parallel
        cp -r ./mindyolo ./train_parallel
        cp -r ./configs ./train_parallel
        cp -r ./tools/*.py ./train_parallel
        cd ./train_parallel || exit

        for((i=0; i<${DEVICE_NUM}; i++))
        do
            start=`expr $i \* $avg`
            end=`expr $start \+ $gap`
            cmdopt=$start"-"$end

            export DEVICE_ID=$i
            export RANK_ID=$i

            echo "start training for rank $RANK_ID, device $DEVICE_ID"
            taskset -c $cmdopt python ./train.py \
                --config=$CONFIG_PATH \
                --device_target=Ascend \
                --is_parallel=True > log$i.txt 2>&1 &
        done

        cd ..
    fi
else
    python ./tools/run.py \
        --config=$CONFIG_PATH \
        --task='train' \
        --device_target=$DEVICE_TRAGET \
        --is_parallel=False > log.txt 2>&1 &
fi
