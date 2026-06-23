# F2HNet: Bridging Local Details and Global Context via Focal-to-Holistic Mixing

**👥 Authors:** Maolin Huang, Pengsheng Huang, Tinghui Li, Shoulin Huang   
**📜 Status:** Under Review at IEEE Transactions on Multimedia (TMM).  
**⚠️ Note:** The manuscript is currently under review. The camera-ready version will be updated upon acceptance.

## Classification on ImageNet-1K  
 The throughputs are measured on single RTX 4090 24G GPU (PyTorch 2.5.1 and CUDA 12.4) with TF32 (TensorFloat-32), and the batch size is set to 64.
| Variants | #Param(M) | FLOPs(G) | Train(img/s) | Infer(img/s) | Acc.1(%) |Acc.5(%)| train log                               | checkpoint| 
|:--------:|:---------:|:--------:|:------------:|:------------:|:--------:|:------:| --------------------------------------- |:------:|
| F2HNet-T | 30        | 4.8      | 588          | 3298         | 83.5     |96.6    | [log](ckpts/cls/Tiny/F2HNet-Tiny.txt)   |[ckpt.pth](https://drive.google.com/file/d/18242jBKFYAXDBDswI1a2zRjQpN6IB1UZ/view?usp=drive_link)|
| F2HNet-S | 45        | 9.1      | 315          | 1644         | 84.6     |97.0    | [log](ckpts/cls/Small/F2HNet-Small.txt) |[ckpt.pth](https://drive.google.com/file/d/114TuwvzpQkEtSc0ixUUOeXpXFoJHIx36/view?usp=drive_link)|
| F2HNet-B | 77        | 15.2     | 222          | 1132         | 85.1     |97.3   | [log](ckpts/cls/Base/F2HNet-Base.txt)   |[ckpt.pth](https://drive.google.com/file/d/1ALaM3D7-lUoNsh_CFtw4ybW_l4fivnBn/view?usp=drive_link)|

## Object Detection with Mask RCNN on COCO 2017

| Schedule | Backbone | #Param(M) | FLOPs(G) | $AP^b$ | $AP^b_{50}$ | $AP^b_{75}$ | $AP^m$ | $AP^m_{50}$ | $AP^m_{75}$ | Train log                                                         | checkpoint |
|:--------:| -------- |:---------:|:--------:|:------:|:-----------:|:-----------:| ------ | ----------- | ----------- | ----------------------------------------------------------------- |-----------|
| 1x       | F2HNet-T | 49.5      | 272      | 45.1   | 67.7        | 49.5        | 40.9   | 64.5        | 43.8        | [log](./ckpts/det/Tiny/mask_rcnn_f2hnet_tiny_1x/log/tiny-log.log) | |
| 3x       | F2HNet-T | 49.5      | 272      | 48.3   | 70.1        | 53.0        | 43.2   | 67.1        | 46.2        | [log](./ckpts/det/Tiny/mask_rcnn_f2hnet_tiny_3x/log/tiny-log.log) |[ckpt.pth](https://drive.google.com/file/d/1Z3Q_ldEvTNbygn2ikPftq07HVDojrlws/view?usp=drive_link) |
| 3x       | F2HNet-S | 64.9      | 363      | 49.5   | 70.8        | 54.3        | 44.0   | 68.1        | 47.4        | [log](./ckpts/det/Small/mask_rcnn_f2hnet_small_3x/log/log.log)    |[ckpt.pth](https://drive.google.com/file/d/1QNf_V7GN-zxr4EgNAVkJaUNE4XVU2E0H/view?usp=drive_link) |
| 3x       | F2HNet-B | 97.0      | 494      | 50.4   | 71.5        | 56.0        | 44.7   | 68.7        | 48.3        | [log](./ckpts/det/base/mask_rcnn_f2hnet_base_3x/log/log.log)      |[ckpt.pth](https://drive.google.com/file/d/1hiTqFvPHhfyIFA-88G8f0G3kdzVyxtSO/view?usp=drive_link) |

## Semantic Sementation with UperNet 160K on ADE20K

| Backbone | #Param(M) | FLOPs(G) |FPS | mIoU | MS mIoU | Train log                                                                              |checkpoint|
| -------- |:---------:|:--------:|:--:|:----:|:-------:| -------------------------------------------------------------------------------------- |----------|
| F2HNet-T | 59        | 955      |23.3| 47.6 | 48.8    | [log](./ckpts/seg/Tiny/f2hnet_tiny_inik_upernet_160k_ade20k_512x512/Train_log/log.log) |[ckpt.pth](https://drive.google.com/file/d/19IXrag2ydRXBxNyxt_nNTGQVI6fIK3vW/view?usp=drive_link) |
| F2HNet-S | 74        | 1054     |18.1| 49.2 | 50.1    | [log](ckpts/seg/Small/f2hnet_small_inik_upernet_160k_ade20k_512x512/Train_log/log.log) |[ckpt.pth](https://drive.google.com/file/d/11XIO8wtl-L9aRaworMbutkn9yl1coi6X/view?usp=drive_link) |
| F2HNet-B | 108       | 1196     |14.9| 49.6 | 50.8    | [log](ckpts/seg/Base/f2hnet_base_inik_upernet_160k_ade20k_512x512/Train_log/log.log)   |[ckpt.pth](https://drive.google.com/file/d/1ZZWL4KHcOoAurqByzITucMNo9MfWpcUR/view?usp=drive_link) |

## Robustness
The evaluation metric for IK/A/R/Sketch/V2 is Top-1 Accuracy(%), and the metric for C is mCE(%).
| Variants | 1K   | C    | A    | R    | Sketch | V2   |
|:--------:|:----:|:----:|:----:|:----:|:------:|:----:|
| F2HNet-T | 83.5 | 48.1 | 34.9 | 48.8 | 35.6   | 72.7 |
| F2HNet-S | 84.6 | 44.4 | 44.2 | 52.5 | 40.4   | 74.8 |
| F2HNet-B | 85.1 | 43.9 | 47.9 | 53.1 | 40.4   | 74.9 |

## Installation:  [classification](classification/README.md),  [Object Detection](detection/README/md), [Semantic Segmentation](segmentation/README.md)

## Cite F2HNet
If you find this repository useful, please give us stars and use the following BibTeX entry for citation.
```

```
