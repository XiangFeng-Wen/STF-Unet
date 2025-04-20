# train_utils/visualize.py
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

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

def save_comparison(pred_mask, gt_mask, raw_input, save_dir, base_name="sample", idx=0):
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
    canvas = Image.new('RGB', (img.width * 3, img.height))
    canvas.paste(img.convert('RGB'), (0, 0))
    canvas.paste(gt_img.convert('RGB'), (img.width, 0))
    canvas.paste(pred_img.convert('RGB'), (img.width * 2, 0))

    # 保存拼图
    canvas.save(os.path.join(save_dir, f"{base_name}_{idx_str}_compare.png"))

    # 可选：分图保存
    img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_img.png"))
    gt_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_gt.png"))
    pred_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_pred.png"))
