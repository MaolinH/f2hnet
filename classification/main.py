import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEmaV2
from tqdm import tqdm

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, save_best_weight, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor, save_best_ema_weight


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--model-ema', action='store_true', )
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', )
    parser.add_argument('--model-ema-eval', action='store_true', help='Using ema to eval during training.')

    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd.')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def main(config):
    cuda_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {cuda_count}")

    if hasattr(config, 'LOCAL_RANK') and config.LOCAL_RANK < cuda_count:
        torch.cuda.set_device(config.LOCAL_RANK)
        print(f"Using CUDA device {config.LOCAL_RANK}")
    else:
        print(f"Invalid LOCAL_RANK: {getattr(config, 'LOCAL_RANK', None)}, defaulting to device 0.")
        if cuda_count > 0:
            torch.cuda.set_device(0)
        else:
            print("No CUDA devices available! Falling back to CPU.")

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    if dist.get_rank() == 0:
        print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if cuda_count >1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, 'flops'):
        flops = model.flops()
    else:
        flops = 0
    if dist.get_rank() == 0:  # any sentense 'if dist.get_rank() == 0:' is to avert repeat writing or printing.
        logger.info(f"number of params: {n_parameters} "
                    f"number of FLOPs: {flops}")

    model.cuda()

    model_ema = None
    # model_ema must be before DDP
    if config.MODEL_EMA:
        model_ema = ModelEmaV2(
            model,
            decay=config.MODEL_EMA_DECAY,
            device='cpu' if config.MODEL_EMA_FORCE_CPU else None,
        )

    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    optimizer = build_optimizer(config, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max5_accuracy = 0.0
    ema_max_accuracy = 0.0
    ema_max5_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                if dist.get_rank() == 0:
                    logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            print(f'auto resuming from {resume_file}')
        else:
            print(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, ema_max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler,
                                                         loss_scaler, model_ema)
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(config, model, logger, data_loader_val)
        return

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            logger.info(
                f'########## Train epoch-{epoch + 1} '
                f'weight_decay: {config.TRAIN.WEIGHT_DECAY} '
                f'ema_decay: {config.MODEL_EMA_DECAY if config.MODEL_EMA else None} ##########')
        lr, train_acc1, grad_norm = train_one_epoch(config, model, criterion, data_loader_train, optimizer,
                                                    epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        if dist.get_rank() == 0:
            logger.info(
                f'-learning rate:{lr:.7f} '
                f'-train_acc1:{train_acc1:.3f} '
                f'-grad_norm:{grad_norm:.2f} ')

        acc1, acc5, loss = validate(config, data_loader_val, model)
        if dist.get_rank() == 0 and acc1 > max_accuracy and epoch > 180:
            save_best_weight(config, model_without_ddp, )
        max_accuracy = max(max_accuracy, acc1)
        max5_accuracy = max(max5_accuracy, acc5)
        if dist.get_rank() == 0:
            logger.info(
                f'--> Val results: '
                f'♣acc1:{acc1:.3f} '
                f'♣acc5:{acc5:.3f} '
                f'♣loss:{loss:.4f} '
                f'♠best_acc1:{max_accuracy:.1f} '
                f'♠best_acc5:{max5_accuracy:.1f}'
            )

        if config.MODEL_EMA_EVAL and config.MODEL_EMA:
            ema_acc1, ema_acc5, ema_loss = validate(config, data_loader_val, model_ema)
            if dist.get_rank() == 0 and ema_acc1 > ema_max_accuracy and epoch > 180:
                save_best_ema_weight(config, model_ema, )
            ema_max_accuracy = max(ema_max_accuracy, ema_acc1)
            ema_max5_accuracy = max(ema_max5_accuracy, ema_acc5)
            if dist.get_rank() == 0:
                logger.info(
                    f'--> Ema val results: '
                    f'♦ema_acc1:{ema_acc1:.3f} '
                    f'♦ema_acc5:{ema_acc5:.3f} '
                    f'♦ema_loss:{ema_loss:.4f} '
                    f'❤best_ema_acc1:{ema_max_accuracy:.1f} '
                    f'❤best_ema_acc5:{ema_max5_accuracy:.1f}'
                )
        # save weight in this epoch:
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer,
                            lr_scheduler, loss_scaler, ema_max_accuracy, model_ema)
        if dist.get_rank() == 0:
            logger.info(
                f'------------------------------------------------------------------------------------------------------------------'
            )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer,
                    epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()

    # get rank
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # create progress bar for each rank
    if rank == 0:
        epoch_bar = tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{config.TRAIN.EPOCHS}',
                         position=rank,
                         leave=True,
                         bar_format='{l_bar}{bar:20}{r_bar}',
                         dynamic_ncols=True)

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            targets_cls = targets.argmax(dim=1) if len(targets.shape) > 1 else targets
        else:
            targets_cls = targets

        with torch.amp.autocast(device_type='cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        acc1, acc5 = accuracy(outputs, targets_cls, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        # Training progress visualization
        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']

        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            memory_used = torch.cuda.max_memory_allocated(gpu_id) / (1024.0 * 1024.0)
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024.0 * 1024.0)
        else:
            memory_used = 0
            memory_reserved = 0

        if dist.is_initialized():
            if rank == 0:  # avg loss and acc1, allocated memory display on rank0
                epoch_bar.update(1)
                epoch_bar.set_postfix({
                    'LR': f'{lr:.7f}',
                    'WD': f'{wd:.5f}',
                    'avg_Loss': f'{loss_meter.avg:.4f}',
                    'avg_grad_norm': f'{norm_meter.avg:.4f}',
                    'avg_Acc1': f'{acc1_meter.avg:.3f}%',
                    f'GPU0': f'{memory_used:.0f}MB',
                })
    # shudown all progress bar
    if rank == 0:
        epoch_bar.close()

    epoch_time = time.time() - start
    return lr, acc1_meter.avg, norm_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    if dist.get_rank() == 0:
        val_bar = tqdm(total=len(data_loader), desc='Validating', position=0,
                       bar_format='{l_bar}{bar:20}{r_bar}')
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if dist.get_rank() == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            val_bar.update(1)
            val_bar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'Acc1': f'{acc1_meter.avg:.3f}%',
                'Acc5': f'{acc5_meter.avg:.3f}',
                'Memory': f'{memory_used:.0f}MB'
            })
    if dist.get_rank() == 0:
        val_bar.close()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(config, model, logger, data_loader=None):
    model.eval()
    if data_loader is None:
        data_loader = []
        img = torch.randn(config.BATCH_SIZE, config.IN_CHANNELS,
                          config.MODEL.FoCoNetPlus.IN_CHANNELS,
                          config.MODEL.FoCoNetPlus.IN_CHANNELS, dtype=torch.float32)  # (B,3,H,W)
        label = torch.randn(config.BATCH_SIZE, config.MODEL.NUM_CLASSES, dtype=torch.float32)  # (B,num_cls)
        data_loader.append([img, label])
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = 1e-6
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.TRAIN.MIN_LR = 1e-5
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())

    main(config)
