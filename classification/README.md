## Installation procedures

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm
pip install yacs
pip install termcolor

```
## Train

```
torchrun --standalone --nproc_per_node=8 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data-path path/to/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 1 \
        --model-ema           \
        --model-ema-decay 0.99992

# Our model train using this command:
torchrun --standalone --nproc_per_node=4 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data-path path/to/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 2 \
        --model-ema           \
        --model-ema-decay 0.99992
```

Test
```
torchrun --standalone --nproc_per_node=1 --master_port 1235 test.py --cfg configs/F2hNet_tiny.yaml --data-path data/imagenet1k --batch-size 128 --test-checkpoint path/to/f2hnet_t.pth

torchrun --standalone --nproc_per_node=1 --master_port 1235 test.py --cfg configs/F2hNet_small.yaml --data-path data/imagenet1k --batch-size 128 --test-checkpoint path/to/f2hnet_s.pth

torchrun --standalone --nproc_per_node=1 --master_port 1235 test.py --cfg configs/F2hNet_base.yaml --data-path data/imagenet1k --batch-size 128 --test-checkpoint path/to/f2hnet_b.pth
```
