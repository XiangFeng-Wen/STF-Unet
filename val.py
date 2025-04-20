import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import STFLSTMUNet
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time


def parse_args():
    parser = argparse.ArgumentParser(description="STF-LSTM-UNet Validation")
    parser.add_argument('--name', default='stf_lstm_unet',
                        help='model name')
    parser.add_argument('--model_path', type=str, default='./save_weights/best_model.pth',
                        help='path to the trained model')
    parser.add_argument('--val_dataset', type=str, default='../../Dataset/myBreaDM/val',
                        help='path to the validation dataset')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    args = parser.parse_args()
    return args


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def main():
    args = parse_args()
    
    # 尝试加载配置文件，如果不存在则使用默认配置
    config = {
        'arch': 'STFLSTMUNet',
        'num_classes': 1,
        'input_channels': 1,
        'deep_supervision': False,
        'input_h': 100,
        'input_w': 100,
        'val_dataset': 'val',
        'img_ext': '.png',
        'mask_ext': '.png',
        'batch_size': 8,
        'num_workers': 4
    }
    
    try:
        with open('./config.yml', 'r') as f:
            config.update(yaml.load(f, Loader=yaml.FullLoader))
    except FileNotFoundError:
        print("Config file not found, using default configuration.")
        # 创建配置文件
        os.makedirs(os.path.dirname('./config.yml'), exist_ok=True)
        with open('./config.yml', 'w') as f:
            yaml.dump(config, f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")
    # 创建模型
    model = STFLSTMUNet(in_channels=config['input_channels'], 
                        num_classes=config['num_classes']+1, 
                        base_c=32)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 导入数据集类
    try:
        from dataset import Dataset
    except ImportError:
        # 如果找不到dataset模块，尝试从上级目录导入
        import sys
        sys.path.append("/home/wxf/project/Segmentation_task/unet")
        from dataset import Dataset

    # 获取验证集图像ID
    val_img_ids = glob(os.path.join(args.val_dataset, 'all_images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    
    # 加载模型权重
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 验证集转换
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 创建验证集
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(args.val_dataset, 'all_images'),
        mask_dir=os.path.join(args.val_dataset, 'all_manual'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 创建评估指标
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', args.name, str(c)), exist_ok=True)
    
    # 开始验证
    print("Starting validation...")
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            
            # 计算模型输出
            start_time = time.time()
            output = model(input)
            if isinstance(output, dict):
                output = output['out']
            gpu_time = time.time() - start_time
            gput.update(gpu_time)

            # 计算IoU和Dice系数
            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            
            # 处理输出
            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            
            # 计算混淆矩阵和准确率
            mat = _fast_hist(target.cpu().numpy().astype(int), output.astype(int), n_class=2)
            acc = np.diag(mat).sum() / mat.sum() if mat.sum() > 0 else 0
            
            # 保存预测结果
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    # 打印评估结果
    print('='*50)
    print(f'Validation Results:')
    print(f'IoU: {iou_avg_meter.avg:.4f}')
    print(f'Dice: {dice_avg_meter.avg:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Average GPU time: {gput.avg:.4f}s')
    print('='*50)

    # 清理GPU缓存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()