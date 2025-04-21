# train_utils/visualize.py
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

import torch

def compute_metrics(pred_logits, target, threshold=0.5, apply_sigmoid=True):
    """
    适用于模型输出的浮点概率图（例如 logits 或 sigmoid 后输出）。
    
    Parameters:
        pred_logits: Tensor [1, H, W] or [H, W]，模型输出（logits 或概率）
        target: Tensor [H, W] or [1, H, W]，ground truth 掩膜
        threshold: 二值化阈值（默认0.5）
        apply_sigmoid: 是否对 pred_logits 应用 sigmoid
        
    Returns:
        dice (float), iou (float)
    """
    smooth = 1e-5

    # 标准化维度
    pred = pred_logits.squeeze().float().detach().cpu()
    target = target.squeeze().float().detach().cpu()

    # 应用 sigmoid
    if apply_sigmoid:
        pred = torch.sigmoid(pred)

    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    # 展平
    pred_np = pred_bin.view(-1).numpy()
    target_np = target_bin.view(-1).numpy()

    # 交并计算
    intersection = (pred_np * target_np).sum()
    union = np.logical_or(pred_np, target_np).sum()

    # IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice
    dice = (2 * intersection + smooth) / (pred_np.sum() + target_np.sum() + smooth)

    return dice, iou


def save_predictions(predictions, save_dir, base_name="pred", threshold=0.5):
    """
    保存一批预测掩膜图像到指定目录
    predictions: Tensor [B, 1, H, W] or [B, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)

    if predictions.ndim == 4:
        predictions = predictions.squeeze(1)  # [B, H, W]

    for i, pred in enumerate(predictions):
        pred_np = pred.detach().cpu().numpy()
        pred_np = (pred_np > threshold) * 255
        pred_img = Image.fromarray(pred_np.astype(np.uint8))
        pred_img.save(os.path.join(save_dir, f"{base_name}_{i:03d}.png"))

def save_comparison(pred_mask, gt_mask, raw_input, save_dir, base_name="sample", idx=0,
                    dice_score=None, iou_score=None):
    """
    保存原始图、GT 掩膜、预测掩膜三张图到一个拼图中，或者分别保存

    Parameters:
        pred_mask: Tensor [1, H, W] or [H, W]
        gt_mask: Tensor [H, W] or None
        raw_input: Tensor [C, H, W] or [1, H, W] (e.g., T×C may已展开)
        save_dir: 保存目录
        base_name: 文件名前缀
        idx: 编号
    """
    os.makedirs(save_dir, exist_ok=True)
    idx_str = f"{idx:03d}"

    # 原始图（取第一个通道或平均）
    if raw_input.ndim == 3:
        if raw_input.shape[0] > 1:
            img_np = raw_input.mean(dim=0).cpu().numpy()
        else:
            img_np = raw_input[0].cpu().numpy()
    else:
        img_np = raw_input.cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5) * 255
    img = Image.fromarray(img_np.astype('uint8')).convert('L')

    # GT mask
    if gt_mask is not None:
        gt_np = gt_mask.cpu().numpy() * 255
        gt_img = Image.fromarray(gt_np.astype('uint8')).convert('L')
    else:
        gt_img = Image.new('L', img.size)

    # Predicted mask
    pred_np = pred_mask.cpu().numpy() * 255
    pred_img = Image.fromarray(pred_np.astype('uint8')).convert('L')

    # 横向拼接
    header = 40      # 标题区
    footer = 30      # 指标信息区
    pad = 20  # 图像之间间隔 20px
    canvas_width = img.width * 3 + pad * 2
    canvas_height = img.height + header + footer
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))  # 留出上方写标题
    canvas.paste(img.convert('RGB'), (0, 40))
    canvas.paste(gt_img.convert('RGB'), (img.width + pad, 40))
    canvas.paste(pred_img.convert('RGB'), (img.width * 2 + pad * 2, 40))

    # 写标题和指标
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    # 标题
    titles = ["Original", "Ground Truth", "Prediction"]
    title_xs = [0, img.width + pad, img.width * 2 + pad * 2]
    for i, title in enumerate(titles):
        draw.text((title_xs[i] + 10, 10), title, fill=(0, 0, 0), font=font)

    # 指标信息
    if dice_score is not None and iou_score is not None:
        print(f"[{idx}] Dice = {dice_score}, IoU = {iou_score}")
        metric_text = f"Dice: {dice_score:.4f} | IoU: {iou_score:.4f}"
        draw.text((10, img.height + header + 5), metric_text, fill=(255, 0, 0), font=font)

    # 保存拼图
    canvas.save(os.path.join(save_dir, f"{base_name}_{idx_str}_compare.png"))

    # 可选：分图保存
    # img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_img.png"))
    # gt_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_gt.png"))
    # pred_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_pred.png"))
