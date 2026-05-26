from models.f2hnet import F2hNet
import sys

def build_model(config):
    if config.MODEL.TYPE == 'F2HNET':
        model = F2hNet(
            in_dim=config.MODEL.F2HNET.IN_CHANNELS,
            patch_size=config.MODEL.F2HNET.PATCH_SIZE ,
            img_size = config.DATA.IMG_SIZE,
            dilation=config.MODEL.F2HNET.DILATION,
            seq_size=config.MODEL.F2HNET.SEQ_SIZE,
            depths=config.MODEL.F2HNET.DEPTHS,
            dims=config.MODEL.F2HNET.DIMS,
            kernel_sizes=config.MODEL.F2HNET.KERNEL_SIZES,
            num_heads = config.MODEL.F2HNET.NUM_HEADS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.F2HNET.MLP_RATIO,
            qkv_bias=config.MODEL.F2HNET.QKV_BIAS,
            qk_scale=config.MODEL.F2HNET.QK_SCALE,
            num_classes=config.MODEL.NUM_CLASSES,
            layer_scale=config.MODEL.F2HNET.INIT_SCALE,
            rpe=config.MODEL.F2HNET.RPE,
        )
    else:
        print(f'Unknown model type: {config.MODEL.TYPE}, please check again.')
        sys.exit(1)
    return model
