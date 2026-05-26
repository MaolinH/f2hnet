CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="mmsegmentation":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}


# bash tools/train.sh configs/foconet/f2hnet_tiny_inik_upernet_160k_ade20k_512x512.py 4 --cfg-options model.backbone.init_cfg.checkpoint='/root/FoCoNet/detection/checkpoints/tiny.pth'