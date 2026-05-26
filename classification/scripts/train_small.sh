
torchrun \
            --standalone \
            --nproc_per_node=8 \
            --master_port 1235 \
      main.py \
      --cfg configs/F2hNet_small.yaml\
      --data/imagenet1k \
      --batch-size 128 \
      --accumulation-steps 1\
      --model-ema           \
      --model-ema-decay 0.99992