import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trn
import torchvision.datasets as dset
from PIL import Image

from models.f2hnet import f2hnet_tiny, f2hnet_small, f2hnet_base


# ==================== 数据集定义 ====================
class NumericImageFolder(Dataset):
    """
    适用于文件夹名为 0,1,2,...,N-1 的数据集。
    文件夹名直接作为类别标签（整数）。
    """
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.samples = []

        # 遍历所有子文件夹（例如 0, 1, 2, ..., 999）
        for class_name in sorted(os.listdir(root), key=lambda x: int(x)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                label = int(class_name)
            except ValueError:
                continue

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm',
                                             '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataloader(root, batch_size=128, num_workers=8):
    """创建标准化的 ImageNet 风格 DataLoader（使用 NumericImageFolder）。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    # dataset = NumericImageFolder(root, transform=transform) # V2
    dataset = dset.ImageFolder(root=root, transform=transform)   # Sketch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, dataset


# ==================== 模型加载工具 ====================
def load_f2hnet(model_variant='base', checkpoint_path=None):
    """
    加载 F2HNet 模型并恢复权重。
    Args:
        model_variant: 模型变体，可选 'tiny', 'small', 'base'。
        checkpoint_path: 权重文件路径；若为 None，则根据变体使用默认路径。
    Returns:
        model: 加载了权重并设为 eval 模式的模型。
    """
    if model_variant == 'tiny':
        model = f2hnet_tiny()
        default_ckpt = 'ckpts/f2hnet_tiny.pth'
    elif model_variant == 'small':
        model = f2hnet_small()
        default_ckpt = 'ckpts/f2hnet_small.pth'
    elif model_variant == 'base':
        model = f2hnet_base()
        default_ckpt = 'ckpts/model.pth'
    else:
        raise ValueError(f"Unknown variant: {model_variant}")

    ckpt_path = checkpoint_path or default_ckpt
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 移除可能存在的 'module.' 前缀（多 GPU 训练后可导致）
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 严格加载，以便立即发现缺失或多余的键
    model.load_state_dict(state_dict, strict=True)
    print(f"Successfully loaded {model_variant} weights from {ckpt_path}")
    return model


# ==================== 评估函数 ====================
@torch.no_grad()
def evaluate(model, dataloader):
    """返回在 dataloader 上的 top-1 准确率（百分比）。"""
    model.eval()
    correct = 0
    total = len(dataloader.dataset)

    for images, targets in dataloader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()

    acc = 100.0 * correct / total
    return acc


# ==================== 主程序 ====================
if __name__ == '__main__':
    # -------------------- 配置 --------------------
    DATA_ROOT = '/disk/data1/Rub_datasets/Sketch/sketch'
    BATCH_SIZE = 256
    NUM_WORKERS = 8

    # 选择模型变体：'tiny', 'small', 'base'
    MODEL_VARIANT = 'base'
    CHECKPOINT_PATH = None   # 使用默认路径，也可手动指定

    # -------------------- 初始化 --------------------
    # 数据
    test_loader, test_dataset = build_dataloader(DATA_ROOT, BATCH_SIZE, NUM_WORKERS)
    print(f"Dataset size: {len(test_dataset)}")

    # 模型
    net = load_f2hnet(MODEL_VARIANT, CHECKPOINT_PATH)
    net.cuda()
    cudnn.benchmark = True   # 固定输入尺寸时加速推理

    # -------------------- 评估 --------------------
    acc = evaluate(net, test_loader)
    print(f"{MODEL_VARIANT.upper()} Top-1 Accuracy: {acc:.2f}%")