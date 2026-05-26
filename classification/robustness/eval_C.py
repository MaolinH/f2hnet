# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.f2hnet import f2hnet_tiny, f2hnet_small, f2hnet_base

# ==================== AlexNet baseline errors ====================
alexnet_err = {
    'gaussian_noise':   0.8864,
    'shot_noise':       0.8944,
    'impulse_noise':    0.9226,
    'defocus_blur':     0.8199,
    'glass_blur':       0.8263,
    'motion_blur':      0.7858,
    'zoom_blur':        0.7984,
    'snow':             0.8668,
    'frost':            0.8267,
    'fog':              0.8194,
    'brightness':       0.5646,
    'contrast':         0.8531,
    'elastic_transform':0.6465,
    'pixelate':         0.7178,
    'jpeg_compression': 0.6065,
    'speckle_noise':    0.8454,
    'gaussian_blur':    0.7255,
    'spatter':          0.8605,
    'saturate':         0.6581
}

# ==================== Configuration ====================
DATA_ROOT = '/disk/data1/Rub_datasets/C_'
BATCH_SIZE = 256
NUM_WORKERS = 8

MODEL_VARIANT = 'base'
CKPT_PATH = 'ckpts/f2hnet_base.pth'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
base_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize(mean, std)
])


# ==================== DDP Initialization ====================
def init_distributed(backend='nccl'):
    """Initialize process group and return local rank."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1

    if rank != -1 and world_size > 1:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0  # single GPU fallback

    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


# ==================== Model Loading (DDP-ready) ====================
def load_model_ddp(variant, ckpt_path, local_rank):
    """Load model and wrap with DDP."""
    if variant == 'tiny':
        model = f2hnet_tiny()
    elif variant == 'small':
        model = f2hnet_small()
    elif variant == 'base':
        model = f2hnet_base()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # Remove 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.cuda(local_rank)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    cudnn.benchmark = True
    return model


# ==================== Data Loading with DistributedSampler ====================
def get_loader(dataset, batch_size, num_workers):
    """Return DataLoader with optional DistributedSampler."""
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),  # no shuffle for distributed
        sampler=sampler, num_workers=num_workers, pin_memory=True
    )
    return loader


# ==================== Evaluation with All-Reduce ====================
@torch.inference_mode()
def evaluate_loader(model, loader, local_rank):
    """Evaluate and return global top-1 accuracy and error."""
    correct = torch.tensor(0.0, device=f'cuda:{local_rank}')
    total = torch.tensor(0.0, device=f'cuda:{local_rank}')
    for images, targets in loader:
        images = images.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum()
        total += targets.size(0)

    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    correct = correct.item()
    total = total.item()
    acc = 100.0 * correct / total
    return acc, 1.0 - correct / total


# ==================== Corruption Evaluation ====================
def show_performance(model, distortion_name, local_rank):
    """Compute mean error over 5 severities for a corruption type."""
    errs = []
    for severity in range(1, 6):
        dataset = dset.ImageFolder(
            root=os.path.join(DATA_ROOT, distortion_name, str(severity)),
            transform=base_transform
        )
        loader = get_loader(dataset, BATCH_SIZE, NUM_WORKERS)
        _, avg_err = evaluate_loader(model, loader, local_rank)
        errs.append(avg_err)

    mean_err = np.mean(errs)
    if local_rank == 0:
        print(f'{distortion_name:20s} | Mean error (unnormalized): {mean_err*100:.2f}%')
    return mean_err


# ==================== Main Worker ====================
def main_worker(local_rank):
    print(f'Worker {local_rank} started')
    model = load_model_ddp(MODEL_VARIANT, CKPT_PATH, local_rank)

    # Distortions list
    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    if local_rank == 0:
        print(f'Evaluating {MODEL_VARIANT.upper()} on ImageNet-C...\n')

    model_mean_errors = []
    for distortion in distortions:
        mean_err = show_performance(model, distortion, local_rank)
        model_mean_errors.append(mean_err)

    # Only rank 0 computes and prints final mCE
    if local_rank == 0:
        mCE_components = []
        for i, distortion in enumerate(distortions):
            normalized = model_mean_errors[i] / alexnet_err[distortion]
            mCE_components.append(normalized)
            print(f'   Normalized CE for {distortion:20s}: {normalized*100:.2f}%')

        mCE = 100 * np.mean(mCE_components)
        print(f'\n=== Standard Normalized mCE (lower is better): {mCE:.4f}% ===')
    if dist.is_initialized():
        dist.barrier()  # synchronize before destroying
    cleanup_distributed()


# ==================== Script Entry Point ====================
if __name__ == '__main__':
    # For torchrun, RANK and WORLD_SIZE are set; LOCAL_RANK is key
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Multi-process mode
        init_distributed()
        main_worker(local_rank)
    else:
        # Single-process fallback
        print('Running in single-GPU mode (DDP not initialized)')
        main_worker(0)