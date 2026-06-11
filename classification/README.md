## Installation procedures

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm
pip install yacs
pip install termcolor
```
Data preparation: ImageNet with the following folder structure.

```
│imagenet1k/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train 
The number of GPUs according to your mechine, and assure: $bs\times ngpu \times accumulation=1024$

```
torchrun --standalone --nproc_per_node=8 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data-path path/to/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 1 \
        --tag ckpts           \ 
        --model-ema           \
        --model-ema-decay 0.99992

# Our model train using this command:
torchrun --standalone --nproc_per_node=4 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data-path path/to/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 2 \
        --tag ckpts        \
        --model-ema           \
        --model-ema-decay 0.99992
```

## Test (The number of GPUs according to your mechine.)
```
# test F2HNet-T
torchrun --standalone --nproc_per_node=8 --master_port 1235 test.py --cfg configs/F2hNet_tiny.yaml --data-path data/imagenet1k --batch-size 64 --test-checkpoint path/to/f2hnet_t.pth
# output result in terminal:
# Test finish!
# Test accuracy: Acc1:83.52% / Acc5:96.48%
# # batch_size 64 throughput 2785

# test F2HNet-S
torchrun --standalone --nproc_per_node=8 --master_port 1235 test.py --cfg configs/F2hNet_small.yaml --data-path data/imagenet1k --batch-size 64 --test-checkpoint path/to/f2hnet_s.pth
# output result in terminal:
# Test finish!
# Test accuracy: Acc1:84.57% / Acc5:96.92%
# batch_size 64 throughput 1461

# test F2HNet-B
torchrun --standalone --nproc_per_node=8 --master_port 1235 test.py --cfg configs/F2hNet_base.yaml --data-path data/imagenet1k --batch-size 64 --test-checkpoint path/to/f2hnet_b.pth
# output result in terminal:
# Test finish!
# Test accuracy: Acc1:85.07% / Acc5:97.19%
# batch_size 64 throughput 861

```
## Test: A simple bash for CPU/single gpu.
You need to overwrite correct ckpt/dataset/model_name on simple_test.py.
```
python simple_test.py
```
