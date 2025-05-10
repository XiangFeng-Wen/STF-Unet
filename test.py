import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser
from src.stf_lstm_unet import STFLSTMUNet
from src import UNet
from my_dataset import DriveDataset
import numpy as np
from datetime import datetime
from train_utils import preprocess_input, evaluate
import transforms as T
from train_utils import merge_images
import cv2

def get_transform(mean=(0.709), std=(0.127)):
    crop_size = 224
    # 只需要调整大小和标准化
    return T.Compose([
        T.RandomResize(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def create_model(model_name, in_channels, num_classes, use_pk_maps=False, time_steps=8):
    if model_name == "unet":
        model = UNet(in_channels=8, num_classes=num_classes)
    elif model_name == "stflstm":
        model = STFLSTMUNet(in_channels=in_channels, num_classes=num_classes,
                            use_pk_maps=use_pk_maps, time_steps=time_steps)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def save_predictions(pred_mask, raw_input, save_dir, base_name="sample", idx=0):
    os.makedirs(save_dir, exist_ok=True)
    idx_str = f"{idx:03d}"
    pred_np = pred_mask.cpu().numpy() * 255
    pred_img = Image.fromarray(pred_np.astype(np.uint8)).convert("L")

    if raw_input.ndim == 3:
        raw_np = raw_input[0].cpu().numpy()
    else:
        raw_np = raw_input.cpu().numpy()
    raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8) * 255
    raw_img = Image.fromarray(raw_np.astype(np.uint8)).convert("L")

    raw_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_img.png"))
    pred_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_pred.png"))

def save_overlay_from_tensor(pred_mask, raw_input, save_dir, patient_id, overlay_color=(0, 255, 0), alpha=0.5):
    """
    将预测掩膜以彩色透明形式叠加在原始图上，并保存

    参数:
        pred_mask: Tensor[H, W]，值为0或1
        raw_input: Tensor[C, H, W] 或 Tensor[H, W]，原始图像
        save_dir: 输出目录
        patient_id: 图像ID（如"001"）
        overlay_color: 叠加颜色（默认绿色）
        alpha: 掩膜透明度（默认0.5）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 原图转numpy（灰度）
    if raw_input.ndim == 3:
        raw_np = raw_input[0].cpu().numpy()
    else:
        raw_np = raw_input.cpu().numpy()
    raw_np = ((raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8) * 255).astype(np.uint8)
    raw_img = cv2.cvtColor(raw_np, cv2.COLOR_GRAY2BGR)

    # 掩膜转numpy
    mask_np = (pred_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask_np = 255 - mask_np

    # 合成图像
    merged = merge_images(raw_img, mask_np, overlay_color, alpha=alpha)

    # 保存
    cv2.imwrite(os.path.join(save_dir, f"unet_{patient_id}.png"), merged)


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
    # img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_img.png"))
    # gt_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_gt.png"))
    # pred_img.save(os.path.join(save_dir, f"{base_name}_{idx_str}_pred.png"))


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, in_channels=1, num_classes=args.num_classes,
                         use_pk_maps=args.use_pk_maps)
    
    model_path = os.path.join(args.model_dir, f"{args.model.lower()}_best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"model_path: {model_path}")

    test_dataset = DriveDataset(root=args.root,
                                 mode="test",
                                 transforms=get_transform(),
                                 use_subtraction=args.use_subtraction,
                                 use_pk_maps=args.use_pk_maps)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True,
                             collate_fn=test_dataset.collate_fn)

    print("🔍 Running inference on test set...")
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = preprocess_input(inputs, model)
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # 保存预测图像
            pred = preds[0][0]
            raw = inputs[0][0] if inputs.ndim == 5 else inputs[0]  # 支持 T×C 和 T×C 展开后的格式
            target = targets[0] if targets is not None else None
            #save_predictions(preds[0][0].cpu(), inputs[0], args.output_dir, base_name=args.model, idx=idx)
            save_overlay_from_tensor(pred, raw, args.output_dir, idx)

    # 如果有GT，计算评估指标
    test_metrics = evaluate(model, test_loader, device=device, num_classes=2)

    print("Test Set Metrics:")
    print(test_metrics["confusion_matrix"])
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"mIoU: {test_metrics['mean_metrics']['miou']:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "stflstm"])
    parser.add_argument("--model-dir", type=str, default="./save_weights", help="路径下存有模型权重")
    parser.add_argument("--root", type=str, default="/home/wxf/project/Dataset/BreaDM", help="测试数据路径")
    parser.add_argument("--output-dir", type=str, default="./output/test_results")
    parser.add_argument("--use-subtraction", action="store_true")
    parser.add_argument("--use-pk-maps", action="store_true")
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    test(args)
