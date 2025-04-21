import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


import transforms


class DriveDataset(Dataset):
    def __init__(self, root: str, mode: str, transforms=None, 
                 sequence_types=None, use_subtraction=False, use_pk_maps=False):
        super(DriveDataset, self).__init__()
        # 根据train参数决定使用哪个数据集
        assert mode in ['train', 'val', 'test'], f"不支持的mode: {mode}"
        self.mode = mode
        self.flag = {
            'train': 'training',
            'val': 'val',
            'test': 'test'
        }[mode]
        # 是否使用PK特征图
        self.use_pk_maps = use_pk_maps
        
        # 默认使用增强序列C1-C8
        if sequence_types is None:
            if use_subtraction:
                sequence_types = [f"SUB{i}" for i in range(1, 9)]
            else:
                sequence_types = [f"VIBRANT+C{i}" for i in range(1, 9)]
        
        self.sequence_types = sequence_types
        print(f"Using sequence types: {self.sequence_types}")
        if use_pk_maps:
            print("Using PK parameter maps (Ktrans, ve, vp)")
            
        # 使用seg目录下的数据
        data_root = os.path.join(root, "seg", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        
        # 获取图像和标签文件路径
        img_dir = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "labels")
        
        # 确保目录存在
        assert os.path.exists(img_dir), f"path '{img_dir}' does not exists."
        assert os.path.exists(mask_dir), f"path '{mask_dir}' does not exists."
        
        # 存储所有图像和对应标签的路径
        self.patient_data = []  # 每个元素是一个字典，包含患者ID、图像路径列表和标签路径
        
        # 遍历所有患者文件夹
        patient_folders = os.listdir(img_dir)
        for patient in patient_folders:
            patient_img_path = os.path.join(img_dir, patient)
            patient_mask_path = os.path.join(mask_dir, patient)
            
            # 确保患者文件夹在images和labels中都存在
            if not os.path.isdir(patient_img_path) or not os.path.isdir(patient_mask_path):
                continue
            
            # 检查该患者是否有所有需要的序列
            all_sequences_exist = True
            for seq_type in self.sequence_types:
                if not os.path.exists(os.path.join(patient_img_path, seq_type)):
                    print(f"Warning: Sequence {seq_type} not found for patient {patient}")
                    all_sequences_exist = False
                    break
            
            if not all_sequences_exist:
                continue
            
            # 如果使用PK特征图，检查是否存在
            pk_maps_exist = True
            pk_maps_path = None
            if self.use_pk_maps:
                pk_maps_path = os.path.join(data_root, "pk_maps", patient)
                if not os.path.exists(pk_maps_path):
                    print(f"Warning: PK maps not found for patient {patient}")
                    pk_maps_exist = False
            
            if self.use_pk_maps and not pk_maps_exist:
                continue
            
            # 获取第一个序列中的所有图像文件名
            first_seq_path = os.path.join(patient_img_path, self.sequence_types[0])
            img_files = [f for f in os.listdir(first_seq_path) if f.endswith('.jpg') or f.endswith('.png')]
            
            for img_file in img_files:
                # 检查所有序列中是否都有这个图像
                all_images_exist = True
                sequence_images = []
                
                for seq_type in self.sequence_types:
                    img_path = os.path.join(patient_img_path, seq_type, img_file)
                    if not os.path.exists(img_path):
                        all_images_exist = False
                        break
                    sequence_images.append(img_path)
                
                if not all_images_exist:
                    continue
                
                # 构建对应的标签路径
                base_name = os.path.splitext(img_file)[0]
                possible_mask_names = [f"{base_name}.png", f"{base_name}.jpg"]
                
                mask_found = False
                mask_path = None
                
                for mask_name in possible_mask_names:
                    # 使用第一个序列对应的标签
                    mask_path = os.path.join(patient_mask_path, self.sequence_types[0], mask_name)
                    if os.path.exists(mask_path):
                        mask_found = True
                        break
                
                if not mask_found:
                    print(f"Warning: No mask found for image {img_file} of patient {patient}")
                    continue
                
                # 添加到数据集
                self.patient_data.append({
                    'patient_id': patient,
                    'image_paths': sequence_images,
                    'mask_path': mask_path,
                    'pk_maps_path': pk_maps_path if self.use_pk_maps else None
                })
        
        # 检查是否找到了图像和标签
        if len(self.patient_data) == 0:
            print(f"Error: No valid image-mask pairs found in {data_root}")
            print(f"Checked sequence types: {self.sequence_types}")
        else:
            print(f"Found {len(self.patient_data)} image-mask pairs for {self.flag} set")

    def __getitem__(self, idx):
        data_item = self.patient_data[idx]
        sequence_images = []
        raw_images = []  # 在这里初始化raw_images列表
        
        # 读取所有时相的图像
        for img_path in data_item['image_paths']:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                img = Image.fromarray(img)
                raw_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                raise
        
        # 应用变换 - 对所有时相图像使用相同的随机变换
        sequence_images = []
        if self.transforms is not None:
            # 对第一张图像和掩码应用变换，获取变换后的掩码
            # 读取掩码（二值图像）
            try:
                mask = Image.open(data_item['mask_path']).convert('L')
                mask_array = np.array(mask) / 255
                mask = Image.fromarray(mask_array.astype(np.uint8))
            except Exception as e:
                print(f"Error loading mask {data_item['mask_path']}: {e}")
                raise
                
            first_img_transformed, mask_transformed = self.transforms(raw_images[0], mask)
            sequence_images.append(first_img_transformed)
            
            # 对其余图像应用相同的变换
            for img in raw_images[1:]:
                img_transformed, _ = self.transforms(img, mask)  # 掩码结果不使用
                sequence_images.append(img_transformed)
        else:
            # 如果没有变换，直接转换为张量
            # 读取掩码（二值图像）
            try:
                mask = Image.open(data_item['mask_path']).convert('L')
                mask_array = np.array(mask) / 255
                mask = Image.fromarray(mask_array.astype(np.uint8))
            except Exception as e:
                print(f"Error loading mask {data_item['mask_path']}: {e}")
                raise
                
            for img in raw_images:
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度
                sequence_images.append(img_tensor)
            mask_transformed = torch.from_numpy(np.array(mask)).long()
        
        # 读取PK特征图（如果启用）- 移到这里，确保sequence_images已经有内容
        if self.use_pk_maps:
            pk_maps = []
            # 获取第一个图像的尺寸，用于创建零填充
            first_img_shape = sequence_images[0].shape
            
            for param_name in ['ktrans', 've', 'vp']:
                pk_path = os.path.join(data_item['pk_maps_path'], f'{param_name}.png')
                try:
                    pk_img = cv2.imread(pk_path, cv2.IMREAD_GRAYSCALE)
                    if pk_img is None:
                        # 如果加载失败，使用零填充
                        pk_img = np.zeros((raw_images[0].height, raw_images[0].width), dtype=np.uint8)
                    pk_img = Image.fromarray(pk_img)
                    
                    # 应用变换
                    if self.transforms is not None:
                        temp_mask = Image.fromarray(np.zeros_like(np.array(pk_img), dtype=np.uint8))
                        pk_transformed, _ = self.transforms(pk_img, temp_mask)
                        pk_maps.append(pk_transformed)
                    else:
                        pk_tensor = torch.from_numpy(np.array(pk_img)).float() / 255.0
                        pk_tensor = pk_tensor.unsqueeze(0)  # 添加通道维度
                        pk_maps.append(pk_tensor)
                except Exception as e:
                    print(f"Error loading PK map {pk_path}: {e}")
                    # 使用零填充，确保形状匹配
                    pk_maps.append(torch.zeros(first_img_shape))
            
            # 将PK特征图添加到序列图像中
            sequence_images.extend(pk_maps)
        
        # 将所有时相图像堆叠成一个张量 [time_steps, channels, height, width]
        sequence_tensor = torch.stack(sequence_images)
        
        return sequence_tensor, mask_transformed

    def __len__(self):
        return len(self.patient_data)

    @staticmethod
    def collate_fn(batch):
        sequences, targets = list(zip(*batch))
        # 序列已经是 [time_steps, channels, height, width]
        # 需要变成 [batch_size, time_steps, channels, height, width]
        batched_sequences = torch.stack(sequences)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_sequences, batched_targets


def cat_list(images, fill_value=0):
    # 处理空列表情况
    if len(images) == 0:
        return torch.zeros((0,))
    
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def visualize_sequence(sequence_tensor, index=0, save_path=None, title='DCE-MRI Sequence'):
    """
    将一个 batch 中的序列样本按时间步可视化为拼图图像。
    
    Args:
        sequence_tensor (torch.Tensor): [B, T, C, H, W]
        index (int): 要展示的 batch 样本索引
        save_path (str or None): 如果提供路径则保存图片
        title (str): 图像标题
    """
    assert sequence_tensor.ndim == 5, "输入应为5维张量 [B, T, C, H, W]"
    seq = sequence_tensor[index]  # [T, C, H, W]
    T = seq.shape[0]

    fig, axs = plt.subplots(1, T, figsize=(T * 2, 2.5))
    if T == 1:
        axs = [axs]

    for i in range(T):
        img = seq[i].squeeze(0).cpu()  # [H, W]
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'T{i+1}', fontsize=8)
        axs[i].axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"图像保存到 {save_path}")
    else:
        plt.show()

def plot_mask_center_tic(sequence_tensor, mask_tensor, index=0, time_interval=1.0, save_path=None):
    """
    提取掩膜区域中心点位置的时间-强度曲线并可视化。

    Args:
        sequence_tensor (torch.Tensor): [B, T, 1, H, W]
        mask_tensor (torch.Tensor): [B, H, W]，二值掩膜
        index (int): 第几个样本
        time_interval (float): 时间间隔（单位：分钟）
        save_path (str or None): 如果给出路径则保存图像
    """
    assert sequence_tensor.ndim == 5 and sequence_tensor.shape[2] == 1, "图像应为 [B, T, 1, H, W]"
    assert mask_tensor.ndim == 3, "掩膜应为 [B, H, W]"

    sequence = sequence_tensor[index].squeeze(1).cpu()  # [T, H, W]
    mask = mask_tensor[index].cpu()                     # [H, W]

    T, H, W = sequence.shape
    mask = (mask > 0).float()

    if mask.sum() == 0:
        print("警告：掩膜为空，无法计算中心点")
        return

    # 获取掩膜内坐标点
    y_coords, x_coords = mask.nonzero(as_tuple=True)

    # 计算中心位置（四舍五入为整数索引）
    y_c = int(y_coords.float().mean().round().item())
    x_c = int(x_coords.float().mean().round().item())

    # 提取中心点的时间曲线
    intensity_curve = sequence[:, y_c, x_c].numpy()
    time_points = np.arange(T) * time_interval

    # 可视化
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))

    axs[0].imshow(sequence[0].numpy(), cmap='gray')
    axs[0].scatter(x_c, y_c, c='red', s=2)  # 小红点
    axs[0].set_title(f"center point: ({x_c}, {y_c})")
    axs[0].axis('off')

    axs[1].plot(time_points, intensity_curve, marker='o', color='blue')
    axs[1].set_title("Time-Intensity Curve (Mask Center)")
    axs[1].set_xlabel("Time (min)")
    axs[1].set_ylabel("Intensity")
    axs[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"图像保存到 {save_path}")
    else:
        plt.show()

def plot_masked_time_intensity_curve(sequence_tensor, mask_tensor, index=0, time_interval=1.0, save_path=None):
    """
    在掩膜区域中寻找最亮点，绘制其在原图中的位置，并输出时间-强度曲线。

    Args:
        sequence_tensor (torch.Tensor): shape [B, T, 1, H, W]
        mask_tensor (torch.Tensor): shape [B, H, W]，取值为0/1或0/255的二值图
        index (int): 选择 batch 中第几个样本
        time_interval (float): 每个时间点的间隔（单位分钟）
        save_path (str or None): 保存图像路径
    """
    assert sequence_tensor.ndim == 5 and sequence_tensor.shape[2] == 1, "输入图像格式应为 [B, T, 1, H, W]"
    assert mask_tensor.ndim == 3, "掩膜格式应为 [B, H, W]"

    sequence = sequence_tensor[index].squeeze(1).cpu()  # [T, H, W]
    mask = mask_tensor[index].cpu()                     # [H, W]

    T, H, W = sequence.shape
    mask = (mask > 0).float()  # 确保为0/1掩膜

    # 计算每个像素在时间维度的最大值
    max_intensity_map = sequence.max(dim=0)[0]  # [H, W]

    # 只保留掩膜内像素
    masked_intensity = max_intensity_map * mask

    if masked_intensity.max() == 0:
        print("警告：掩膜内无非零像素，无法提取时间曲线")
        return

    # 找到掩膜区域中亮度最大的位置
    y, x = torch.nonzero(masked_intensity == masked_intensity.max(), as_tuple=True)
    y, x = int(y[0]), int(x[0])

    # 提取该像素的时间曲线
    intensity_curve = sequence[:, y, x].numpy()
    time_points = np.arange(T) * time_interval

    # 可视化
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))

    axs[0].imshow(sequence[0].numpy(), cmap='gray')
    axs[0].scatter(x, y, c='red', s=1)
    axs[0].set_title(f"point set: ({x}, {y})")
    axs[0].axis('off')

    axs[1].plot(time_points, intensity_curve, marker='o', color='blue')
    axs[1].set_title("Time-Intensity Curve (Mask)")
    axs[1].set_xlabel("Time (min)")
    axs[1].set_ylabel("Intensity")
    axs[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"图像保存到 {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # 测试数据集加载
    from torch.utils.data import DataLoader
    import transforms as T
    
    # 创建简单的变换
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.709), std=(0.127))
    ])
    
    # 测试增强序列和减影序列
    print("\nTesting with enhancement sequences (VIBRANT+C1 to VIBRANT+C8)")
    dataset_enhancement = DriveDataset(
        root='/home/wxf/project/Dataset/BreaDM', 
        mode='train', 
        transforms=transform,
        sequence_types=[f"VIBRANT+C{i}" for i in range(1, 9)]
    )
    
    print("\nTesting with subtraction sequences (SUB1 to SUB8)")
    dataset_subtraction = DriveDataset(
        root='/home/wxf/project/Dataset/BreaDM', 
        mode='train', 
        transforms=transform,
        sequence_types=[f"SUB{i}" for i in range(1, 9)]
    )
    
    # 测试默认参数
    print("\nTesting with default parameters (should use VIBRANT+C1 to VIBRANT+C8)")
    dataset_default = DriveDataset(
        root='/home/wxf/project/Dataset/BreaDM', 
        mode='train',  
        transforms=transform
    )
    
    # 测试减影序列的默认参数
    print("\nTesting with default subtraction sequences")
    dataset_default_sub = DriveDataset(
        root='/home/wxf/project/Dataset/BreaDM', 
        mode='train', 
        transforms=transform,
        use_subtraction=True
    )
    
    # 选择一个非空数据集进行DataLoader测试
    test_dataset = None
    for dataset in [dataset_enhancement, dataset_subtraction, dataset_default, dataset_default_sub]:
        if len(dataset) > 0:
            test_dataset = dataset
            break
    
    if test_dataset is not None:
        dataloader = DataLoader(
            test_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=test_dataset.collate_fn
        )
        
        # 打印数据集大小
        print(f"\nSelected dataset size: {len(test_dataset)}")
        sample_count = 0
        max_samples = 10
        output_dir = '/home/wxf/project/Segmentation_task/STF-Unet/output/dataset_test'
        os.makedirs(output_dir, exist_ok=True)  # 递归创建目录，exist_ok=True 避免目录已存在时报错
        # 获取一个批次并打印形状
        try:
            for sequences, masks in dataloader:
                batch_size = sequences.size(0)

                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break

                    save_path = os.path.join(output_dir, f"seq_{sample_count+1:03d}.png")
                    visualize_sequence(sequences, index=i, save_path=save_path, title=f"Sample {sample_count+1} - Sequence T=8")
                    save_path = os.path.join(output_dir, f"tic_{sample_count+1:03d}.png")
                    plot_mask_center_tic(sequences, masks, index=i, time_interval=1, save_path=save_path)
                    sample_count += 1

                if sample_count >= max_samples:
                    break
        except Exception as e:
            print(f"Error loading batch: {e}")