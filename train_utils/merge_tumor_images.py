#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并减影图和掩码图，生成带掩码的肿瘤切片序列

用法:
    python merge_tumor_images.py --patient_id 001 --output_dir ./output

参数:
    --patient_id: 病人编号
    --output_dir: 输出目录
    --subtraction_dir: 减影图目录 (默认: Dataset/myBreaDM/training/all_images)
    --mask_dir: 掩码图目录 (默认: Dataset/myBreaDM/training/all_manual)
    --overlay_color: 掩码覆盖颜色 (默认: 255,0,0 - 红色)
    --overlay_alpha: 掩码透明度 (默认: 0.5)
    --border_only: 是否只显示边界 (默认: False)
    --border_thickness: 边界厚度 (默认: 2)
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='合并减影图和掩码图，生成带掩码的肿瘤切片序列')
    parser.add_argument('--patient_id', type=str, required=True, help='病人编号')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--subtraction_dir', type=str, 
                        default='/home/wxf/project/Dataset/myBreaDM/training/all_images', 
                        help='减影图目录')
    parser.add_argument('--mask_dir', type=str, 
                        default='/home/wxf/project/Dataset/myBreaDM/training/all_manual', 
                        help='掩码图目录')
    parser.add_argument('--overlay_color', type=str, default='255,0,0', 
                        help='掩码覆盖颜色 (R,G,B)')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, 
                        help='掩码透明度 (0-1)')
    parser.add_argument('--border_only', action='store_true', 
                        help='是否只显示边界')
    parser.add_argument('--border_thickness', type=int, default=2, 
                        help='边界厚度')
    return parser.parse_args()

def find_patient_images(directory, patient_id, extension='.jpg'):
    """查找指定病人编号的图像文件"""
    pattern = os.path.join(directory, f"{patient_id}*{extension}")
    return sorted(glob(pattern))

def load_image(image_path):
    """加载图像文件"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return img

def load_mask(mask_path):
    """加载掩码图像文件"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法加载掩码图像: {mask_path}")
    # 二值化掩码图像
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def create_overlay_mask(mask, color, alpha=0.5):
    """创建彩色半透明掩码"""
    # 创建彩色掩码
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask > 0] = color
    
    # 创建透明度掩码
    alpha_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
    alpha_mask[mask > 0] = alpha
    
    return color_mask, alpha_mask

def create_border_mask(mask, color, thickness=2):
    """创建边界掩码"""
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建空白图像用于绘制轮廓
    border_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 绘制轮廓
    cv2.drawContours(border_mask, contours, -1, color, thickness)
    
    return border_mask

def merge_images(image, mask, color, alpha=0.5, border_only=False, border_thickness=2):
    """合并原始图像和掩码"""
    # 确保图像是彩色的
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 解析颜色
    if isinstance(color, str):
        color = tuple(map(int, color.split(',')))
    
    # 创建合并后的图像副本
    merged = image.copy()
    
    if border_only:
        # 创建边界掩码并合并
        border_mask = create_border_mask(mask, color, border_thickness)
        # 将边界掩码添加到原始图像
        merged = cv2.addWeighted(merged, 1.0, border_mask, 1.0, 0)
    else:
        # 创建彩色半透明掩码
        color_mask, alpha_mask = create_overlay_mask(mask, color, alpha)
        
        # 合并图像和掩码
        for c in range(3):  # 对每个颜色通道
            merged[:,:,c] = image[:,:,c] * (1 - alpha_mask) + color_mask[:,:,c] * alpha_mask
    
    return merged.astype(np.uint8)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找病人的减影图和掩码图
    subtraction_images = find_patient_images(args.subtraction_dir, args.patient_id, '.jpg')
    mask_images = find_patient_images(args.mask_dir, args.patient_id, '.png')
    
    if not subtraction_images:
        print(f"未找到病人 {args.patient_id} 的减影图")
        return
    
    if not mask_images:
        print(f"未找到病人 {args.patient_id} 的掩码图")
        return
    
    print(f"找到 {len(subtraction_images)} 张减影图和 {len(mask_images)} 张掩码图")
    
    # 解析颜色
    color = tuple(map(int, args.overlay_color.split(',')))
    
    # 处理每对图像
    for i, (sub_path, mask_path) in enumerate(zip(subtraction_images, mask_images)):
        try:
            # 加载图像
            sub_img = load_image(sub_path)
            mask_img = load_mask(mask_path)
            
            # 确保图像尺寸一致
            if sub_img.shape[:2] != mask_img.shape[:2]:
                mask_img = cv2.resize(mask_img, (sub_img.shape[1], sub_img.shape[0]))
            
            # 合并图像
            merged_img = merge_images(
                sub_img, mask_img, color, 
                alpha=args.overlay_alpha, 
                border_only=args.border_only, 
                border_thickness=args.border_thickness
            )
            
            # 保存结果
            output_filename = f"{args.patient_id}_{i+1:03d}_merged.png"
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, merged_img)
            
            print(f"已保存: {output_path}")
            
        except Exception as e:
            print(f"处理图像 {sub_path} 和 {mask_path} 时出错: {e}")
    
    print(f"处理完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()