import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from src import STFLSTMUNet
from train_utils import compute_metrics, EarlyStopping, train_one_epoch, evaluate, create_lr_scheduler, preprocess_input
import transforms as T


# 预处理类，在训练时的数据增强包括随机缩放、水平翻转、垂直翻转、随机裁剪，然后转为Tensor和标准化
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485), std=(0.229)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

# 验证，调整大小到crop_size，然后同样的转换
class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485), std=(0.229)):
        self.transforms = T.Compose([
            T.RandomResize(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.709), std=(0.127)):
    # 根据BreaDM数据集特点调整参数
    base_size = 256
    crop_size = 224

    if train:
        # 增强训练数据集的变换
        trans = [
            T.RandomResize(int(0.5 * base_size), int(1.2 * base_size)),
            T.RandomHorizontalFlip(0.5),  # 水平翻转（镜像）
            T.RandomVerticalFlip(0.5),    # 垂直翻转
            T.RandomRotation(degrees=30),  # 随机旋转±30度
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        return T.Compose(trans)
    else:
        # 验证集只需要调整大小和标准化
        return T.Compose([
            T.RandomResize(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

def create_model(model_type, num_classes, in_channels=1, use_pk_maps=False, time_steps=8):
    if model_type == 'stflstm':
        from src import STFLSTMUNet
        return STFLSTMUNet(
            in_channels=in_channels, 
            num_classes=num_classes, 
            time_steps=time_steps,
            use_pk_maps=use_pk_maps
        )
    elif model_type == 'unet':
        from src import UNet
        if (use_pk_maps == True) :
            return UNet(in_channels=8+3, num_classes=num_classes)
        else:
            return UNet(in_channels=8, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def parse_args():
    parser = argparse.ArgumentParser(description="STF-LSTM-UNet Training")
    parser.add_argument('--model', default='stflstm', choices=['stflstm', 'unet'], help='choose model type')
    parser.add_argument('--data-path', default='/home/wxf/project/Dataset/BreaDM', help='dataset path')
    parser.add_argument('--num-classes', default=1, type=int, help='number of classes')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--aux', action='store_true', help='use auxiliary loss')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--save-best', action='store_true', default=True, help='only save best dice weights')
    parser.add_argument('--amp', action='store_true', help='use torch.cuda.amp for mixed precision training')
    parser.add_argument('--tf32', action='store_true', help='enable TF32 mode for faster training on Ampere GPUs')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--silent', action='store_true', help='do not save results file')
    parser.add_argument('--use-pk-maps', action='store_true', help='use PK parameter maps')
    parser.add_argument('--generate-pk-maps', action='store_true', help='generate PK parameter maps before training')
    parser.add_argument('--use-subtraction', action='store_true', help='use subtraction images instead of enhanced images')
    parser.add_argument('--test-only', action='store_true', help='仅进行模型测试，不训练')

    return parser.parse_args()


def main(args):
    print(torch.__version__)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device training.")

    # 启用TF32加速（需PyTorch 1.12+）
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 tensor cores")

    # 优化cudnn配置
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cudnn.deterministic = False  # 允许非确定性优化

    # 数据加载优化
    num_workers = args.workers  # 使用自定义workers数

    batch_size = args.batch_size
    # segmentation num_classes + background
    num_classes = args.num_classes + 1

    # 使用compute_mean_std.py计算的BreaDM数据集均值和标准差
    mean = (0.709)
    std = (0.127)

    # 根据是否使用PK参数图，设置路径标识
    tag_suffix = "_pk" if args.use_pk_maps else ""

    # 用来保存训练以及验证过程中信息
    results_file = None
    if not args.silent:
        results_file = "./output/{}_results_{}{}.txt".format(
            args.model,
            datetime.datetime.now().strftime("%m%d-%H%M"),
            tag_suffix
        )
        # 创建输出目录
        os.makedirs("./output", exist_ok=True)

    # 如果需要，生成PK特征图
    if args.generate_pk_maps:
        print("Generating PK parameter maps...")
        from pk_fitting import generate_pk_maps_for_dataset
        generate_pk_maps_for_dataset(args.data_path)
        print("PK parameter maps generation completed")

    early_stopper = EarlyStopping(patience=20, verbose=True)

    # 导入数据集类 - 使用我们为BreaDM数据集创建的自定义DriveDataset
    from my_dataset import DriveDataset

    # 创建数据集实例
    train_dataset = DriveDataset(args.data_path,
                                 mode='train',
                                 transforms=get_transform(train=True, mean=mean, std=std),
                                 use_subtraction=args.use_subtraction,
                                 use_pk_maps=args.use_pk_maps)

    val_dataset = DriveDataset(args.data_path,
                               mode='val',
                               transforms=get_transform(train=False, mean=mean, std=std),
                               use_subtraction=args.use_subtraction,
                               use_pk_maps=args.use_pk_maps)

    # 智能配置num_workers
    cpu_count = os.cpu_count()
    # 根据经验，num_workers通常设置为CPU核心数的0.5-1倍较为合适
    suggested_workers = max(1, min(int(cpu_count * 0.75), batch_size * 2))
    # 考虑用户指定的workers数量
    num_workers = min([suggested_workers, args.workers if args.workers > 0 else float('inf')])

    print(f"Using {num_workers} workers for data loading, batch_size: {batch_size}")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,  # 锁页内存加速传输
                              prefetch_factor=2,  # 预加载批次，避免内存过载
                              persistent_workers=True,  # 保持worker进程存活
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)

    # 确定输入通道数
    in_channels = 1
    
    # 创建模型实例
    model = create_model(
        model_type=args.model,
        num_classes=num_classes, 
        in_channels=in_channels,
        use_pk_maps=args.use_pk_maps,
        time_steps=len(train_dataset.sequence_types) if hasattr(train_dataset, 'sequence_types') else 8
    )

    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,                    # 学习率
        betas=(0.9, 0.999),            # AdamW的动量参数
        weight_decay=args.weight_decay, # 权重衰减
        eps=1e-8,                       # 数值稳定性参数
        fused=True                      # 使用融合实现（如果可用）
    )

    # 混合精度训练
    scaler = GradScaler() if args.amp else None
    if args.amp:
        print(f"Mixed precision training enabled (AMP mode)")

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    from train_utils import create_lr_scheduler, train_one_epoch, evaluate
        
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 创建保存权重的目录
    save_dir = "./save_weights"
    os.makedirs(save_dir, exist_ok=True)
    
    best_dice = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.test_only:
            break
        # 调用train_one_epoch函数
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        
        # 评估模型
        eval_metrics = evaluate(model, val_loader, device=device, num_classes=num_classes)
        dice = eval_metrics["dice"]
        confmat = eval_metrics["confusion_matrix"]
        global_acc = eval_metrics["global_accuracy"]
        class_metrics = eval_metrics["class_metrics"]
        mean_metrics = eval_metrics["mean_metrics"]
        
        # 打印验证结果
        val_info = confmat
        print(val_info)
        print(f"Dice coefficient: {dice:.4f}")
        print(f"Global accuracy: {global_acc:.4f}")
        print(f"Mean IoU: {mean_metrics['miou']:.4f}")
        print(f"Mean precision: {mean_metrics['mprecision']:.4f}")
        print(f"Mean recall: {mean_metrics['mrecall']:.4f}")

        # 写入结果文件
        if results_file:
            with open(results_file, "a") as f:
                # 记录每个epoch的训练信息
                info = f"[epoch: {epoch}]\n"
                info += f"train_loss: {mean_loss:.4f}\n"
                info += f"lr: {lr:.6f}\n"
                info += f"dice: {dice:.4f}\n"
                info += f"global_acc: {global_acc:.4f}\n"
                info += f"mean_iou: {mean_metrics['miou']:.4f}\n"
                info += f"mean_precision: {mean_metrics['mprecision']:.4f}\n"
                info += f"mean_recall: {mean_metrics['mrecall']:.4f}\n"
                info += f"{val_info}\n\n"
                f.write(info)

        # 保存权重
        save_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        model_name = args.model.lower()

        if args.save_best:
            # 每轮保存 latest
            latest_path = os.path.join(save_dir, f"{model_name}_latest_model{tag_suffix}.pth")
            torch.save(save_file, latest_path)

            # 保存最优模型
            if best_dice < dice:
                best_path = os.path.join(save_dir, f"{model_name}_best_model{tag_suffix}.pth")
                torch.save(save_file, best_path)
                best_dice = dice
                print(f"✅ New best model saved at epoch {epoch}, Dice = {dice:.4f}")
        else:
            # 每轮保存
            epoch_path = os.path.join(save_dir, f"{model_name}_epoch{epoch}{tag_suffix}.pth")
            torch.save(save_file, epoch_path)

        # ✅ 检查早停
        if early_stopper.step(dice):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 计算总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    print("Start evaluating best model on test set...")

    model_name = args.model.lower()

    # 加载最佳模型
    best_model_path = os.path.join(save_dir, f"{model_name}_best_model.pth")
    assert os.path.exists(best_model_path), f"{model_name}_best_model.pth not found!"
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 创建测试集
    test_dataset = DriveDataset(args.data_path,
                                 mode='test', 
                                 transforms=get_transform(train=False, mean=mean, std=std),
                                 use_subtraction=args.use_subtraction,
                                 use_pk_maps=args.use_pk_maps)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             collate_fn=test_dataset.collate_fn)

    # 推理评估
    from train_utils.visualize import save_comparison  # 如果你已有保存工具
    print("Running inference on test set...")

    test_save_dir = "./output/test_results{tag_suffix}"
    os.makedirs(test_save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = preprocess_input(inputs, model)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # 保存预测图像
            pred = 1 - preds[0][0]
            raw = inputs[0][0] if inputs.ndim == 5 else inputs[0]  # 支持 T×C 和 T×C 展开后的格式
            target = targets[0] if targets is not None else None
            dice_val, iou_val = compute_metrics(pred, target) if target is not None else (None, None)
            
            save_comparison(pred, target, raw, test_save_dir, base_name=model_name, idx=idx,
                            dice_score=dice_val, iou_score=iou_val)
    
    # 如果有GT，计算评估指标
    print("Evaluating predictions on test set...")
    test_metrics = evaluate(model, test_loader, device=device, num_classes=num_classes)

    print("Test Set Metrics:")
    print(test_metrics["confusion_matrix"])
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"mIoU: {test_metrics['mean_metrics']['miou']:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)