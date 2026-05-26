# More Codes are Coming!!!!!

## Classification on Imagen-1K

| Variants | #Param(M) | FLOPs(G) | Train(img/s) | Infer(img/s) | Top-1(%) | train log                               |
|:--------:|:---------:|:--------:|:------------:|:------------:|:--------:| --------------------------------------- |
| F2HNet-T | 30        | 4.8      | 588          | 1784         | 83.5     | [log](ckpts/cls/Tiny/F2HNet-Tiny.txt)   |
| F2HNet-S | 45        | 9.1      | 315          | 912          | 84.6     | [log](ckpts/cls/Small/F2HNet-Small.txt) |
| F2HNet-B | 77        | 15.2     | 222          | 648          | 85.1     | [log](ckpts/cls/Base/F2HNet-Base.txt)   |

## Object Detection with Mask RCNN on COCO 2017

| Schedule | Backbone | #Param(M) | FLOPs(G) | $AP^b$ | $AP^b_{50}$ | $AP^b_{75}$ | $AP^m$ | $AP^m_{50}$ | $AP^m_{75}$ | Train log                                                         |
|:--------:| -------- |:---------:|:--------:|:------:|:-----------:|:-----------:| ------ | ----------- | ----------- | ----------------------------------------------------------------- |
| 1x       | F2HNet-T | 49.5      | 272      | 45.1   | 67.7        | 49.5        | 40.9   | 64.5        | 43.8        | [log](./ckpts/det/Tiny/mask_rcnn_f2hnet_tiny_1x/log/tiny-log.log) |
| 3x       | F2HNet-T | 49.5      | 272      | 48.3   | 70.1        | 53.0        | 43.2   | 67.1        | 46.2        | [log](./ckpts/det/Tiny/mask_rcnn_f2hnet_tiny_3x/log/tiny-log.log) |
| 3x       | F2HNet-S | 64.9      | 363      | 49.5   | 70.8        | 54.3        | 44.0   | 68.1        | 47.4        | [log](./ckpts/det/Small/mask_rcnn_f2hnet_small_3x/log/log.log)    |
| 3x       | F2HNet-B | 97.0      | 494      | 50.4   | 71.5        | 56.0        | 44.7   | 68.7        | 48.3        | [log](./ckpts/det/Base/mask_rcnn_f2hnet_base_3x/log/log.log)      |

## Semantic Sementation with UperNet 160K on ADE20K

| Backbone | #Param(M) | FLOPs(G) | mIoU | MS mIoU | Train log                                                                              |
| -------- |:---------:|:--------:|:----:|:-------:| -------------------------------------------------------------------------------------- |
| F2HNet-T | 59        | 955      | 47.6 | 48.8    | [log](./ckpts/seg/Tiny/f2hnet_tiny_inik_upernet_160k_ade20k_512x512/Train_log/log.log) |
| F2HNet-S | 74        | 1054     | 49.2 | 50.1    | [log](ckpts/seg/Small/f2hnet_small_inik_upernet_160k_ade20k_512x512/Train_log/log.log) |
| F2HNet-B | 108       | 1196     | 49.6 | 50.8    | [log](ckpts/seg/Base/f2hnet_base_inik_upernet_160k_ade20k_512x512/Train_log/log.log)   |

## Robustness

| Variants | 1K   | C    | A    | R    | Sketch | V2   |
|:--------:|:----:|:----:|:----:|:----:|:------:|:----:|
| F2HNet-T | 83.5 | 48.1 | 34.9 | 48.8 | 35.6   | 72.7 |
| F2HNet-S | 84.6 | 44.4 | 44.2 | 52.5 | 40.4   | 74.8 |
| F2HNet-B | 85.1 | 43.9 | 47.9 | 53.1 | 40.4   | 74.9 |

## Installation:  [classification](classification/README.md),  [Object Detection](detection/README/md), [Semantic Segmentation](segmentation/README.md)

## Train bash

```
torchrun --standalone --nproc_per_node=8 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 1 \
        --model-ema           \
        --model-ema-decay 0.99992

# Our model train using this command:
torchrun --standalone --nproc_per_node=4 --master_port 1235 \
        main.py \
        --cfg configs/F2hNet_tiny.yaml\
        --data/imagenet1k \
        --batch-size 128 \
        --accumulation-steps 2 \
        --model-ema           \
        --model-ema-decay 0.99992
```
