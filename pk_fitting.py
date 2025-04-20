import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class ToftsModelFitter:
    """
    扩展Tofts模型拟合器，用于DCE-MRI数据分析
    """
    
    def __init__(self, time_points=None, device=None, aif_method='population'):
        # 设置计算设备
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 默认时间点设置（假设8个时相，每个时相间隔约1分钟）
        if time_points is None:
            self.time_points = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device=self.device)
        else:
            self.time_points = torch.tensor(time_points, dtype=torch.float32, device=self.device)
            
        # 设置AIF方法，默认使用标准人口AIF模型
        self.aif_method = aif_method
    
    def population_aif(self, t, dose=0.1):
        """
        标准人口AIF模型 (Parker模型)
        
        Args:
            t: 时间点张量
            dose: 对比剂剂量 (mmol/kg)，默认0.1
            
        Returns:
            AIF曲线张量
        """
        # Parker模型参数
        a1, a2 = 3.99, 4.78
        m1, m2 = 0.144, 0.0111
        
        # 计算Parker模型
        aif = dose * (a1 * torch.exp(-m1 * t) + a2 * torch.exp(-m2 * t))
        
        return aif
    
    def modified_aif(self, t):
        """
        改进的动脉输入函数 (AIF)，更符合乳腺DCE-MRI特性
        使用双指数模型，更好地模拟对比剂在血液中的分布
        """
        # 双指数模型参数
        a1, a2 = 3.99, 4.78
        m1, m2 = 0.144, 0.0111
        return a1 * torch.exp(-m1 * t) + a2 * torch.exp(-m2 * t)
    
    def aif(self, t):
        """
        动脉输入函数 (AIF)，支持多种方法
        """
        if self.aif_method == 'population':
            # 使用标准人口AIF模型
            return self.population_aif(t)
        
        elif self.aif_method == 'auto':
            # 使用自动检测的AIF
            if hasattr(self, 'aif_concentration'):
                # 需要对时间点进行插值，因为t可能与self.time_points不同
                if t.shape[0] == self.time_points.shape[0] and torch.allclose(t, self.time_points):
                    # 如果时间点完全相同，直接返回AIF曲线
                    return self.aif_concentration
                else:
                    # 否则进行插值
                    t_np = t.cpu().numpy()
                    time_points_np = self.time_points.cpu().numpy()
                    aif_curve_np = self.aif_concentration.cpu().numpy()
                    
                    # 使用线性插值
                    from scipy.interpolate import interp1d
                    f = interp1d(time_points_np, aif_curve_np, kind='linear', bounds_error=False, fill_value='extrapolate')
                    interpolated_aif = f(t_np)
                    
                    return torch.tensor(interpolated_aif, device=self.device, dtype=torch.float32)
            else:
                # 如果没有自动检测的AIF曲线，使用改进的双指数模型代替
                return self.modified_aif(t)
        
        elif self.aif_method == 'modified':
            # 使用改进的双指数模型
            return self.modified_aif(t)
        
        else:
            raise ValueError(f"不支持的AIF方法: {self.aif_method}")
    
    def get_auto_detected_aif(self, images_tensor, tissue_mask):
        """
        基于时间导数最大值自动检测AIF
        
        Args:
            images_tensor: 形状为 [time_steps, height, width] 的图像张量
            tissue_mask: 组织掩码
            
        Returns:
            AIF曲线张量和检测位置
        """
        # 转换为numpy进行处理
        images_np = images_tensor.cpu().numpy()
        
        # 计算每个体素的时间导数最大值，找变化最快的点
        diff = np.diff(images_np, axis=0)  # 差分
        peak_diff = np.max(diff, axis=0)
        
        # 只在组织区域内查找
        masked_peak_diff = peak_diff * tissue_mask.cpu().numpy()
        
        # 找最大位置（疑似血管）
        artery_pos = np.unravel_index(np.argmax(masked_peak_diff), masked_peak_diff.shape)
        x, y = artery_pos
        
        # 使用该点的时间曲线作为参考AIF
        aif_curve = images_np[:, x, y]
        
        # 将AIF信息保存为类属性，以便后续使用
        self.aif_position = (x, y)
        self.aif_curve = torch.tensor(aif_curve, device=self.device, dtype=torch.float32)
        self.aif_concentration = aif_concentration
        
        return self.aif_concentration, self.aif_position
    
    def convert_signal_to_concentration(self, signal_curves, baseline_indices=None):
        """
        将信号强度曲线转换为造影剂浓度曲线
        
        Args:
            signal_curves: 形状为 [num_pixels, time_steps] 的信号强度曲线
            baseline_indices: 基线时间点的索引，默认为第一个时间点
            
        Returns:
            形状为 [num_pixels, time_steps] 的造影剂浓度曲线
        """
        # 如果没有指定基线时间点，使用第一个时间点作为基线
        if baseline_indices is None:
            baseline_indices = [0]
        
        # 计算基线信号
        baseline_signal = torch.mean(signal_curves[:, baseline_indices], dim=1, keepdim=True)
        
        # 计算相对增强度 (S - S0) / S0
        relative_enhancement = (signal_curves - baseline_signal) / (baseline_signal + 1e-6)
        
        # 线性转换为浓度（假设信号与浓度成正比）
        concentration = relative_enhancement
        
        return concentration
    
    def preprocess_images(self, images):
        """
        预处理图像，包括归一化和组织分割
        
        Args:
            images: 形状为 [time_steps, height, width] 的图像数组
            
        Returns:
            归一化后的图像张量和组织掩码
        """
        time_steps, height, width = images.shape
        
        # 转换为张量并归一化
        images_tensor = torch.tensor(images, dtype=torch.float32, device=self.device)
        
        # 归一化到 [0, 1] 范围
        images_tensor = images_tensor / 255.0
        
        # 创建组织掩码 - 使用第一个时间点的图像
        first_image = images[0]
        
        # 使用简单阈值分割识别组织区域，而不是二值化
        # 假设背景较暗，组织较亮
        threshold = np.mean(first_image) * 0.15
        tissue_mask = first_image > threshold
        
        # 应用形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        
        # 转换为张量
        tissue_mask_tensor = torch.tensor(tissue_mask, dtype=torch.bool, device=self.device)
        
        return images_tensor, tissue_mask_tensor
    
    def extended_tofts_model_batch(self, t, Ktrans, ve, vp):
        """
        批量计算扩展Tofts模型
        """
        batch_size = Ktrans.shape[0]
        time_steps = t.shape[0]
        
        # 计算AIF
        aif_values = self.aif(t)  # [time_steps]
        
        # 准备卷积计算
        dt = 0.01
        max_time = t[-1].item()
        t_conv = torch.arange(0, max_time, dt, device=self.device, dtype=torch.float32)  # 确保使用float32类型
        aif_conv = self.aif(t_conv)  # [conv_steps]
        
        # 初始化结果张量
        result = torch.zeros((batch_size, time_steps), device=self.device, dtype=torch.float32)  # 确保使用float32类型
        
        # 对每个时间点计算卷积
        for i, ti in enumerate(t):
            # 只考虑小于当前时间的卷积点
            mask = t_conv < ti
            if not mask.any():
                continue
                
            t_valid = t_conv[mask]  # [valid_steps]
            aif_valid = aif_conv[mask]  # [valid_steps]
            
            # 计算指数项 [batch_size, valid_steps]
            exp_term = torch.exp(-Ktrans.view(-1, 1) * (ti - t_valid.view(1, -1)) / ve.view(-1, 1))
            
            # 计算卷积和
            conv_sum = torch.sum(aif_valid.view(1, -1) * exp_term, dim=1) * dt  # [batch_size]
            
            # 计算该时间点的结果
            result[:, i] = vp * aif_values[i] + Ktrans * conv_sum

        return result
    
    def fit_volume_gpu(self, subtraction_images, output_dir=None, debug_output_dir=None):
        """
        使用GPU对整个体积的减影图像进行拟合
        
        Args:
            subtraction_images: 形状为 [time_steps, height, width] 的减影图像数组
            output_dir: 输出目录，用于保存参数图
            debug_output_dir: 调试输出目录
            
        Returns:
            形状为 [3, height, width] 的参数图，分别是Ktrans, ve, vp
        """
        time_steps, height, width = subtraction_images.shape
        
        print("开始预处理图像...")
        start_time = time.time()
        # 预处理图像
        images_tensor, tissue_mask = self.preprocess_images(subtraction_images)
        print(f"预处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 初始化参数图
        param_maps = torch.zeros((3, height, width), dtype=torch.float32, device=self.device)
        
        # 批处理大小
        batch_size = 1024  # 可以根据GPU内存调整
        
        # 将图像重塑为像素批次
        pixels = images_tensor.permute(1, 2, 0).reshape(-1, time_steps)  # [height*width, time_steps]
        
        # 创建掩码，只拟合有效组织区域
        pixel_mask = tissue_mask.reshape(-1)
        valid_pixels = pixels[pixel_mask]  # [num_valid, time_steps]
        
        print(f"总像素数: {height*width}, 有效像素数: {valid_pixels.shape[0]}")
        
        # 直接使用时间浓度曲线，不进行转换
        
        # 保存一些像素的时间曲线用于调试
        if debug_output_dir is not None:
            # 随机选择10个有效像素
            num_samples = min(10, valid_pixels.shape[0])
            sample_indices = torch.randperm(valid_pixels.shape[0])[:num_samples]
            sample_pixels = valid_pixels[sample_indices]
            
            plt.figure(figsize=(10, 6))
            for i in range(num_samples):
                plt.plot(self.time_points.cpu().numpy(), sample_pixels[i].cpu().numpy(), 
                         marker='o', label=f'Pixel {i+1}')
                plt.xlabel('Time (min)')
                plt.ylabel('Signal Intensity')
                plt.title('Sample Pixel Time Curves')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(debug_output_dir, "sample_time_curves.png"))
                plt.close()
        
        # 改进的初始参数猜测 - 更适合乳腺组织，确保使用float32类型
        initial_ktrans = torch.full((valid_pixels.shape[0],), 0.05, device=self.device, dtype=torch.float32)
        initial_ve = torch.full((valid_pixels.shape[0],), 0.1, device=self.device, dtype=torch.float32)
        initial_vp = torch.full((valid_pixels.shape[0],), 0.01, device=self.device, dtype=torch.float32)
        
        # 参数需要梯度
        ktrans = initial_ktrans.clone().detach().requires_grad_(True)
        ve = initial_ve.clone().detach().requires_grad_(True)
        vp = initial_vp.clone().detach().requires_grad_(True)
        
        # 优化器 - 使用更小的学习率
        optimizer = torch.optim.Adam([ktrans, ve, vp], lr=0.005)
        
        # 更严格的参数约束函数 - 限制在生理学合理范围内
        def constrain_params():
            with torch.no_grad():
                ktrans.clamp_(0.0, 1.0)  # 更严格的上限
                ve.clamp_(0.001, 0.5)    # 更严格的上限
                vp.clamp_(0.0, 0.2)      # 更严格的上限
        
        # 训练循环
        num_epochs = 100
        losses = []
        
        print("开始拟合Tofts模型...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 分批处理
            total_loss = 0.0
            num_batches = (valid_pixels.shape[0] + batch_size - 1) // batch_size
            
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, valid_pixels.shape[0])
                
                # 直接使用信号强度曲线
                batch_pixels = valid_pixels[start_idx:end_idx]  # [batch, time_steps]
                batch_ktrans = ktrans[start_idx:end_idx]
                batch_ve = ve[start_idx:end_idx]
                batch_vp = vp[start_idx:end_idx]
                
                # 前向传播
                predicted = self.extended_tofts_model_batch(self.time_points, batch_ktrans, batch_ve, batch_vp)
                
                # 计算损失
                loss = F.mse_loss(predicted, batch_pixels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 应用约束
                constrain_params()
                
                total_loss += loss.item()
            
            # 记录损失
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            
            # 打印进度 - 与fit_volume保持一致
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        print(f"拟合完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 绘制损失曲线
        if debug_output_dir is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            plt.savefig(os.path.join(debug_output_dir, "training_loss.png"))
            plt.close()
        
        # 将优化后的参数填充到参数图中
        param_tensor = torch.zeros((3, height*width), dtype=torch.float32, device=self.device)
        param_tensor[0, pixel_mask] = ktrans.detach()
        param_tensor[1, pixel_mask] = ve.detach()
        param_tensor[2, pixel_mask] = vp.detach()
        
        # 重塑为原始图像尺寸
        param_maps = param_tensor.reshape(3, height, width)
        
        # 后处理参数图 - 与fit_volume保持一致，不进行额外处理
        param_maps_np = param_maps.cpu().numpy()
        
        # 如果提供了输出目录，保存参数图
        if output_dir is not None:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存参数图
            param_names = ['ktrans', 've', 'vp']
            param_cmaps = ['hot', 'cool', 'spring']  # 使用不同的颜色映射
            
            for i, name in enumerate(param_names):
                # 获取参数图
                param_map = param_maps_np[i]
                
                # 归一化参数图以便可视化
                if np.max(param_map) > 0:
                    # 使用百分位数裁剪，避免极值影响
                    p_min, p_max = np.percentile(param_map[param_map > 0], [1, 99])
                    norm_map = np.clip(param_map, p_min, p_max)
                    norm_map = ((norm_map - p_min) / (p_max - p_min) * 255).astype(np.uint8)
                else:
                    norm_map = np.zeros_like(param_map, dtype=np.uint8)
                    
                # 保存参数图
                output_file = os.path.join(output_dir, f'{name}.png')
                cv2.imwrite(output_file, norm_map)
                
                # 创建热图可视化 - 使用不同的颜色映射
                plt.figure(figsize=(8, 6))
                plt.imshow(param_map, cmap=param_cmaps[i])
                plt.colorbar(label=name)
                plt.title(f'{name.upper()} Parameter Map')
                plt.savefig(os.path.join(output_dir, f'{name}_heatmap.png'))
                plt.close()
                
                # 保存原始参数值
                np.save(os.path.join(output_dir, f'{name}_raw.npy'), param_map)
            
            # 创建整合热力图
            self.create_combined_heatmap(param_maps_np, output_dir)
        
        return param_maps_np
    
    def fit_volume(self, subtraction_images, output_dir):
        """
        对整个体积的减影图像进行拟合
        
        Args:
            subtraction_images: 形状为 [time_steps, height, width] 的减影图像数组
            output_dir: 输出目录，用于保存参数图
            
        Returns:
            形状为 [3, height, width] 的参数图，分别是Ktrans, ve, vp
        """
        time_steps, height, width = subtraction_images.shape
        
        print("开始预处理图像...")
        start_time = time.time()
        # 预处理图像
        images_tensor, tissue_mask = self.preprocess_images(subtraction_images)
        print(f"预处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 初始化参数图
        param_maps = torch.zeros((3, height, width), dtype=torch.float32, device=self.device)
        
        # 批处理大小
        batch_size = 1024  # 可以根据GPU内存调整
        
        # 将图像重塑为像素批次
        pixels = images_tensor.permute(1, 2, 0).reshape(-1, time_steps)  # [height*width, time_steps]
        
        # 创建掩码，只拟合有效组织区域
        pixel_mask = tissue_mask.reshape(-1)
        valid_pixels = pixels[pixel_mask]  # [num_valid, time_steps]
        
        print(f"总像素数: {height*width}, 有效像素数: {valid_pixels.shape[0]}")
        
        # 直接使用时间浓度曲线，不进行转换
        # 改进的初始参数猜测 - 更适合乳腺组织，确保使用float32类型
        initial_ktrans = torch.full((valid_pixels.shape[0],), 0.05, device=self.device, dtype=torch.float32)
        initial_ve = torch.full((valid_pixels.shape[0],), 0.1, device=self.device, dtype=torch.float32)
        initial_vp = torch.full((valid_pixels.shape[0],), 0.01, device=self.device, dtype=torch.float32)
        
        # 参数需要梯度
        ktrans = initial_ktrans.clone().detach().requires_grad_(True)
        ve = initial_ve.clone().detach().requires_grad_(True)
        vp = initial_vp.clone().detach().requires_grad_(True)
        
        # 优化器 - 使用更小的学习率
        optimizer = torch.optim.Adam([ktrans, ve, vp], lr=0.005)
        
        # 更严格的参数约束函数 - 限制在生理学合理范围内
        def constrain_params():
            with torch.no_grad():
                ktrans.clamp_(0.0, 1.0)  # 更严格的上限
                ve.clamp_(0.001, 0.5)    # 更严格的上限
                vp.clamp_(0.0, 0.2)      # 更严格的上限
        
        # 训练循环
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # 分批处理
            total_loss = 0.0
            num_batches = (valid_pixels.shape[0] + batch_size - 1) // batch_size
            
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, valid_pixels.shape[0])
                
                # 直接使用信号强度曲线
                batch_pixels = valid_pixels[start_idx:end_idx]  # [batch, time_steps]
                batch_ktrans = ktrans[start_idx:end_idx]
                batch_ve = ve[start_idx:end_idx]
                batch_vp = vp[start_idx:end_idx]
                
                # 前向传播
                predicted = self.extended_tofts_model_batch(self.time_points, batch_ktrans, batch_ve, batch_vp)
                
                # 计算损失
                loss = F.mse_loss(predicted, batch_pixels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 应用约束
                constrain_params()
                
                total_loss += loss.item()
            
            # 每10个epoch打印一次进度
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # 将优化后的参数填充到参数图中
        param_tensor = torch.zeros((3, height*width), dtype=torch.float32, device=self.device)
        param_tensor[0, pixel_mask] = ktrans.detach()
        param_tensor[1, pixel_mask] = ve.detach()
        param_tensor[2, pixel_mask] = vp.detach()
        
        # 重塑为原始图像尺寸
        param_maps = param_tensor.reshape(3, height, width)
        
        # 后处理参数图
        param_maps_np = param_maps.cpu().numpy()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存参数图
        param_names = ['ktrans', 've', 'vp']
        param_cmaps = ['hot', 'cool', 'spring']  # 使用不同的颜色映射
        
        for i, name in enumerate(param_names):
            # 获取参数图
            param_map = param_maps_np[i]
            
            # 归一化参数图以便可视化
            if np.max(param_map) > 0:
                # 使用百分位数裁剪，避免极值影响
                p_min, p_max = np.percentile(param_map[param_map > 0], [1, 99])
                norm_map = np.clip(param_map, p_min, p_max)
                norm_map = ((norm_map - p_min) / (p_max - p_min) * 255).astype(np.uint8)
            else:
                norm_map = np.zeros_like(param_map, dtype=np.uint8)
                
            # 保存参数图
            output_file = os.path.join(output_dir, f'{name}.png')
            cv2.imwrite(output_file, norm_map)
            
            # 创建热图可视化 - 使用不同的颜色映射
            plt.figure(figsize=(8, 6))
            plt.imshow(param_map, cmap=param_cmaps[i])
            plt.colorbar(label=name)
            plt.title(f'{name.upper()} Parameter Map')
            plt.savefig(os.path.join(output_dir, f'{name}_heatmap.png'))
            plt.close()
            
            # 保存原始参数值
            np.save(os.path.join(output_dir, f'{name}_raw.npy'), param_map)
        
        # 创建整合热力图
        self.create_combined_heatmap(param_maps_np, output_dir)
        
        return param_maps_np
    
    def create_combined_heatmap(self, param_maps, output_dir):
        """
        创建Ktrans、ve、vp的整合热力图
        
        Args:
            param_maps: 形状为 [3, height, width] 的参数图数组
            output_dir: 输出目录
        """
        # 获取参数图
        ktrans = param_maps[0]
        ve = param_maps[1]
        vp = param_maps[2]
        
        # 创建RGB图像
        height, width = ktrans.shape
        combined_map = np.zeros((height, width, 3), dtype=np.float32)
        
        # 归一化每个参数图
        for i, param_map in enumerate([ktrans, ve, vp]):
            if np.max(param_map) > 0:
                p_min, p_max = np.percentile(param_map[param_map > 0], [1, 99])
                norm_map = np.clip(param_map, p_min, p_max)
                norm_map = (norm_map - p_min) / (p_max - p_min)
                combined_map[:, :, i] = norm_map
        
        # 保存整合热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(combined_map)
        plt.title('Combined Parameter Map (R:Ktrans, G:Ve, B:Vp)')
        plt.savefig(os.path.join(output_dir, 'combined_heatmap.png'))
        plt.close()
        
        # 保存为图像文件
        combined_map_uint8 = (combined_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'combined_map.png'), cv2.cvtColor(combined_map_uint8, cv2.COLOR_RGB2BGR))


def process_patient(patient_path, output_base_dir):
    """
    处理单个患者的DCE-MRI数据
    
    Args:
        patient_path: 患者数据路径
        output_base_dir: 输出基础目录
    """
    # 获取患者ID
    patient_id = os.path.basename(patient_path)
    print(f"处理患者: {patient_id}")
    
    # 创建输出目录
    output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载减影图像
    subtraction_images = []
    image_paths = []
    
    for i in range(1, 9):
        sub_folder = os.path.join(patient_path, f'SUB{i}')
        if not os.path.exists(sub_folder):
            print(f"警告: {sub_folder} 不存在")
            continue
            
        # 获取该文件夹中的所有图像
        image_files = [f for f in os.listdir(sub_folder) if f.endswith('.jpg') or f.endswith('.png')]
        
        if not image_files:
            print(f"警告: {sub_folder} 中没有找到图像")
            continue
            
        # 只处理第一张图像（简化处理）
        img_path = os.path.join(sub_folder, image_files[0])
        
        # 加载图像为灰度图，但不进行二值化
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
            
        subtraction_images.append(img)
    
    if not subtraction_images:
        print(f"错误: 在 {patient_path} 中没有找到有效的减影图像")
        return
    
    print("使用以下图像进行拟合:")
    for i, path in enumerate(image_paths):
        print(f"  时相 {i+1}: {path}")
        
    # 转换为numpy数组
    subtraction_images = np.array(subtraction_images)
    
    # 归一化图像
    subtraction_images = subtraction_images / 255.0
    
    # 创建Tofts模型拟合器，默认使用标准人口AIF模型
    fitter = ToftsModelFitter(aif_method='population')
    
    # 处理患者数据
    pk_maps = fitter.fit_volume_gpu(subtraction_images, output_dir)
    
    print(f"患者 {patient_id} 的PK参数图已保存到: {output_dir}")


def process_dataset(dataset_path, split='training'):
    """
    处理整个数据集
    
    Args:
        dataset_path: 数据集路径
        split: 数据集分割（training, validation, testing）
    """
    # 设置输入和输出路径
    images_dir = os.path.join(dataset_path, 'seg', split, 'images')
    output_base_dir = os.path.join(dataset_path, 'seg', split, 'pk_maps')
    
    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取所有患者
    patients = [p for p in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, p))]
    
    print(f"找到 {len(patients)} 个患者")
    
    # 处理每个患者
    for patient in patients:
        patient_path = os.path.join(images_dir, patient)
        process_patient(patient_path, output_base_dir)


def generate_pk_maps_for_dataset(dataset_path, splits=None):
    """
    为数据集生成PK参数图
    
    Args:
        dataset_path: 数据集路径
        splits: 要处理的数据集分割列表，默认为['training', 'validation', 'testing']
    
    Returns:
        生成的PK参数图的路径字典
    """
    if splits is None:
        splits = ['training', 'val', 'test']
    
    output_paths = {}
    
    for split in splits:
        print(f"为{split}集生成PK参数图...")
        output_dir = os.path.join(dataset_path, 'seg', split, 'pk_maps')
        process_dataset(dataset_path, split)
        output_paths[split] = output_dir
        print(f"{split}集的PK参数图已保存到: {output_dir}")
    
    return output_paths


if __name__ == "__main__":
    # 设置数据集路径
    dataset_path = '/home/wxf/project/Dataset/BreaDM'
    
    # 处理所有数据集分割
    generate_pk_maps_for_dataset(dataset_path)