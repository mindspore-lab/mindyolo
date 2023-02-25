if [ $# != 4 ]
then
    echo "Usage: bash run_test.sh [CONFIG_PATH] [DEVICE_TRAGET] [DEVICE_ID|CUDA_VISIBLE_DEVICES] [WEIGHT]"
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
WEIGHT=$4
DEVICE_NUM=1

PARALLEL=0
if [ $DEVICE_NUM -gt 1 ]; then
    PARALLEL=1
fi

if [ $DEVICE_TRAGET == 'Ascend' ]; then
    echo "Run Ascend"
    if [ $DEVICE_NUM == 1 ]; then
        DEVICE_ID=$3
        export DEVICE_NUM=1
        export DEVICE_ID=$DEVICE_ID
        export RANK_SIZE=1
        export RANK_ID=$DEVICE_ID
    else
        echo "error: Ascend device num not equal 1."
        exit 1
    fi
elif [ $DEVICE_TRAGET == 'GPU' ]; then
    echo "Run GPU"
    export CUDA_VISIBLE_DEVICES=$3
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

python ./tools/eval.py \
    --config=$CONFIG_PATH \
    --task='eval' \
    --device_target=$DEVICE_TRAGET \
    --weight=$WEIGHT  > log.txt 2>&1 &
