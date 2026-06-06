torchrun --standalone --nproc_per_node=8 \
            --master_port 1235 \
              test.py \
                --cfg configs/F2hNet_small.yaml \
                --data-path data/imagenet1k \
                --batch-size 128 \
                --test-checkpoint path/to/f2hnet_s.pth
