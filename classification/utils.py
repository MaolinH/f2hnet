
import os
import torch
import torch.distributed as dist
# from tensorboardX import SummaryWriter


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, model_ema=None):
    print(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    max_accuracy = 0.0
    ema_max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'ema_max_accuracy' in checkpoint:
            ema_max_accuracy = checkpoint['ema_max_accuracy']

    if config.MODEL_EMA:
        if 'model_ema' in checkpoint:
            model_ema.load_state_dict(checkpoint['model_ema'])
        else:
            model_ema.module.load_state_dict(checkpoint['model'])
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, ema_max_accuracy

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                    ema_max_accuracy=None, model_ema=None):
    save_state = {
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    if model_ema is not None:
        save_state['model_ema'] = model_ema.state_dict()
    if ema_max_accuracy is not None:
        save_state['ema_max_accuracy'] = ema_max_accuracy

    save_path = os.path.join(config.OUTPUT, f'ckpt.pth')
    print(f"{save_path} saving ~")
    torch.save(save_state, save_path)
    print(f"{save_path} saved #")

def save_best_weight(config,  model, ):
    save_state = {'model': model.state_dict(),
                  }
    save_path = os.path.join(config.OUTPUT, f'model.pth')
    print(f"{save_path} saving ~")
    torch.save(save_state, save_path)
    print(f"{save_path} saved #")


def save_best_ema_weight(config, model_ema, ):
    save_state = {
                  'model': model_ema.module.state_dict(),
                  }
    save_path = os.path.join(config.OUTPUT, f'model_ema.pth')
    print(f"{save_path} saving ~")
    torch.save(save_state, save_path)
    print(f"{save_path} saved #")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    # print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# class TensorboardLogger(object):
#     def __init__(self, log_dir):
#         self.writer = SummaryWriter(logdir=log_dir)
#         self.step = 0

#     def set_step(self, step=None):
#         if step is not None:
#             self.step = step
#         else:
#             self.step += 1

#     def update(self, head='scalar', step=None, **kwargs):
#         for k, v in kwargs.items():
#             if v is None:
#                 continue
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

#     def flush(self):
#         self.writer.flush()