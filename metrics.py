import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def iou_score(output, target):
    """
    计算IoU分数和Dice系数
    
    Args:
        output: 模型输出，形状为[B, C, H, W]
        target: 目标掩码，形状为[B, C, H, W]
        
    Returns:
        iou: IoU分数
        dice: Dice系数
    """
    smooth = 1e-5
    
    if isinstance(output, dict):
        output = output['out']
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)
    
    return iou, dice


class AverageMeter(object):
    """
    计算并存储平均值和当前值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count