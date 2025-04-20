import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的短边缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 转换为numpy数组
        image_np = np.array(image)
        target_np = np.array(target)
        
        # 检查图像维度
        if len(image_np.shape) == 2:  # 灰度图像
            h, w = image_np.shape
            has_channel = False
        else:  # 彩色图像
            h, w = image_np.shape[:2]
            has_channel = True
        
        if h < self.size or w < self.size:
            # 如果图像尺寸小于裁剪尺寸，则进行填充
            pad_h = max(self.size - h, 0)
            pad_w = max(self.size - w, 0)
            
            if has_channel:
                # 彩色图像，需要考虑通道维度
                image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            else:
                # 灰度图像，没有通道维度
                image_np = np.pad(image_np, ((0, pad_h), (0, pad_w)), mode='constant')
            
            # 目标掩码总是单通道的
            if len(target_np.shape) == 2:
                target_np = np.pad(target_np, ((0, pad_h), (0, pad_w)), mode='constant')
            else:
                target_np = np.pad(target_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
            # 更新尺寸
            if has_channel:
                h, w = image_np.shape[:2]
            else:
                h, w = image_np.shape
        
        # 随机选择裁剪的起始位置
        h_start = random.randint(0, h - self.size)
        w_start = random.randint(0, w - self.size)
        
        # 裁剪图像和目标
        if has_channel:
            image_crop = image_np[h_start:h_start + self.size, w_start:w_start + self.size, :]
        else:
            image_crop = image_np[h_start:h_start + self.size, w_start:w_start + self.size]
        
        if len(target_np.shape) == 2:
            target_crop = target_np[h_start:h_start + self.size, w_start:w_start + self.size]
        else:
            target_crop = target_np[h_start:h_start + self.size, w_start:w_start + self.size, :]
        
        # 转换回PIL图像
        image_crop = Image.fromarray(image_crop.astype(np.uint8))
        target_crop = Image.fromarray(target_crop.astype(np.uint8))
        
        return image_crop, target_crop


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean=(0.485), std=(0.229)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, target):
        """
        对图像和目标掩码应用相同的随机旋转
        
        Args:
            img (PIL Image): 输入图像
            target (PIL Image): 目标掩码
            
        Returns:
            PIL Image: 旋转后的图像
            PIL Image: 旋转后的掩码
        """
        if random.random() < 0.5:  # 50%的概率应用旋转
            angle = random.uniform(-self.degrees, self.degrees)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            target = target.rotate(angle, resample=Image.NEAREST, expand=False)
        return img, target
