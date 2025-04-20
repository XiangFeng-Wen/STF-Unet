import torch
from torch import nn
import numpy as np
import torch.distributed as dist
import time
from collections import defaultdict, deque
import datetime

def preprocess_input(inputs, model):
    input_format = getattr(model, "input_format", "time_sequence")

    if input_format == "flat_channels":
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B, T * C, H, W)
    elif input_format == "average_frame":
        inputs = inputs.mean(dim=1)  # [B, C, H, W]
    elif input_format == "time_sequence":
        pass  # 保持原始形状 [B, T, C, H, W]
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    return inputs

# 混淆矩阵类，用于计算分割评估指标
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 转换为一维tensor，并且为每个类别计算混淆矩阵
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}'
        ).format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

# Dice系数计算类
class DiceCoefficient(object):
    def __init__(self, num_classes=2, ignore_index=None):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if isinstance(pred, dict):
            pred = pred['out']
            
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask
            target = target * mask

        pred = pred.view(-1)
        target = target.view(-1)

        # 计算每个类别的Dice系数
        dice_per_class = []
        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            if union > 0:
                dice = (2.0 * intersection) / union
            else:
                dice = torch.tensor(1.0, device=pred.device)  # 如果没有该类别，则Dice为1
                
            dice_per_class.append(dice)
        
        # 计算平均Dice系数
        dice_tensor = torch.stack(dice_per_class)
        if self.cumulative_dice is None:
            self.cumulative_dice = dice_tensor
            self.count = 1
        else:
            self.cumulative_dice += dice_tensor
            self.count += 1

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0)
        return self.cumulative_dice / self.count

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)

    @property
    def value(self):
        if self.cumulative_dice is None:
            return torch.tensor(0.0)
        return self.compute().mean()

    def reset(self):
        self.cumulative_dice = None
        self.count = 0

# 平滑值类，用于记录训练过程中的指标
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

# 指标记录器类，用于记录训练过程中的各种指标
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# 损失函数
def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            from .dice_coefficient_loss import dice_loss, build_target
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

# 评估函数
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes) # 用于记录每个类别的混淆矩阵数据
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = MetricLogger(delimiter="  ") # 用于每 100 个 batch 打印一次日志
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image = preprocess_input(image, model)
            image = image.to(device)
            target = target.to(device)

            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        mat = confmat.mat.cpu().numpy()  # 获取混淆矩阵数值

    eps = 1e-6  # 防止除零的小量
    total = mat.sum()
    class_metrics = []

    # 计算全局准确率
    global_accuracy = np.diag(mat).sum() / total if total != 0 else 0.0

    # 遍历每个类别计算指标
    for cls_idx in range(num_classes):
        tp = mat[cls_idx, cls_idx]
        fp = mat[:, cls_idx].sum() - tp
        fn = mat[cls_idx, :].sum() - tp
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        
        class_metrics.append({
            "precision": precision,
            "recall": recall,
            "iou": iou,
        })

    # 计算平均指标
    mean_metrics = {
        "mprecision": np.mean([m["precision"] for m in class_metrics]),
        "mrecall": np.mean([m["recall"] for m in class_metrics]),
        "miou": np.mean([m["iou"] for m in class_metrics]),
    }

    return {
        "dice": dice.value.item(),
        "confusion_matrix": confmat,
        "global_accuracy": global_accuracy,
        "class_metrics": class_metrics,
        "mean_metrics": mean_metrics
    }

# 训练一个epoch的函数
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                   lr_scheduler=None, print_freq=10, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")  # 注意这里不需要utils.前缀
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = preprocess_input(image, model)
        image = image.to(device)
        target = target.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            # 确保模型接收正确形状的输入
            # 数据集返回的image形状为[batch_size, time_steps, channels, height, width]
            # 模型期望的输入形状为[batch_size, channels, height, width]
            # 我们不需要在这里修改形状，因为模型的forward方法应该已经处理了这种情况
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

# 创建学习率调度器
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)