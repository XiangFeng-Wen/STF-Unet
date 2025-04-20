import os
import torch
import numpy as np
from collections import OrderedDict


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


def save_checkpoint(state, filename='checkpoint.pth'):
    """
    保存检查点
    """
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    加载检查点
    """
    if not os.path.exists(checkpoint):
        return
    
    state = torch.load(checkpoint)
    model.load_state_dict(state['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    
    print(f"加载检查点 '{checkpoint}' (epoch {state['epoch']})")
    
    return state


def convert_state_dict(state_dict):
    """
    移除'module.'前缀
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # 移除'module.'
        new_state_dict[name] = v
    return new_state_dict