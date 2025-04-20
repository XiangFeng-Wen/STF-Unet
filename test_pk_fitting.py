import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import matplotlib

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的字体列表
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

# 尝试使用系统可用的中文字体
def setup_chinese_font():
    try:
        # 尝试导入fonttools来检查系统字体
        from matplotlib.font_manager import FontManager
        
        # 获取系统字体列表
        font_manager = FontManager()
        font_names = [f.name for f in font_manager.ttflist]
        
        # 常见的中文字体名称
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
        
        # 查找系统中可用的中文字体
        for font in chinese_fonts:
            if font in font_names:
                matplotlib.rcParams['font.sans-serif'].insert(0, font)
                print(f"使用中文字体: {font}")
                return True
        
        print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")
        return False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return False

# 在程序开始时设置字体
setup_chinese_font()

class ToftsModelFitter:
    """
    扩展Tofts模型拟合器，用于从DCE-MRI减影图像中提取药代动力学参数
    使用GPU加速计算，并增强预处理和噪声抑制
    """
    def __init__(self, time_points=None, device=None, aif_method='auto'):
        # 设置计算设备
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 默认时间点设置（假设8个时相，每个时相间隔约1分钟）
        if time_points is None:
            self.time_points = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device=self.device)
        else:
            self.time_points = torch.tensor(time_points, dtype=torch.float32, device=self.device)
            
        # 设置AIF方法
        self.aif_method = aif_method
        print(f"使用AIF方法: {self.aif_method}")
    
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
            if hasattr(self, 'aif_curve'):
                # 需要对时间点进行插值，因为t可能与self.time_points不同
                if t.shape[0] == self.time_points.shape[0] and torch.allclose(t, self.time_points):
                    # 如果时间点完全相同，直接返回AIF曲线
                    return self.aif_curve
                else:
                    # 否则进行插值
                    t_np = t.cpu().numpy()
                    time_points_np = self.time_points.cpu().numpy()
                    aif_curve_np = self.aif_curve.cpu().numpy()
                    
                    # 使用线性插值
                    from scipy.interpolate import interp1d
                    f = interp1d(time_points_np, aif_curve_np, kind='linear', bounds_error=False, fill_value='extrapolate')
                    interpolated_aif = f(t_np)
                    
                    return torch.tensor(interpolated_aif, device=self.device, dtype=torch.float32)
            else:
                print("警告: 未找到自动检测的AIF曲线，使用改进的双指数模型代替")
                return self.modified_aif(t)
        
        elif self.aif_method == 'modified':
            # 使用改进的双指数模型
            return self.modified_aif(t)
        
        else:
            raise ValueError(f"不支持的AIF方法: {self.aif_method}")
    
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
    
    def get_auto_detected_aif(self, images_tensor, tissue_mask, debug_output_dir=None):
        """
        基于时间导数最大值自动检测AIF
        
        Args:
            images_tensor: 形状为 [time_steps, height, width] 的图像张量
            tissue_mask: 组织掩码
            debug_output_dir: 调试输出目录
            
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
        print(f"自动检测血管位置：({x}, {y})")
        
        # 使用该点的时间曲线作为参考AIF
        aif_curve = images_np[:, x, y]
        
        # 保存AIF曲线和位置信息
        if debug_output_dir is not None:
            # 获取最大增强图用于标记
            max_image = np.max(images_np, axis=0)
            max_image_norm = (max_image * 255).astype(np.uint8)
            
            # 绘制AIF曲线
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_points.cpu().numpy(), aif_curve, 'ro-', linewidth=2)
            plt.xlabel('Time (min)')
            plt.ylabel('Signal Intensity')
            plt.title('Detected AIF Curve')
            plt.grid(True)
            plt.savefig(os.path.join(debug_output_dir, "detected_aif_curve.png"))
            plt.close()
            
            # 在最大增强图上标记AIF位置
            aif_marker = max_image_norm.copy()
            # 在AIF位置画一个红色圆圈
            cv2.circle(aif_marker, (y, x), 5, 255, 2)
            cv2.imwrite(os.path.join(debug_output_dir, "aif_location.png"), aif_marker)
            
            # 保存时间导数最大值图
            plt.figure(figsize=(8, 6))
            plt.imshow(masked_peak_diff, cmap='hot')
            plt.colorbar(label='Max Time Derivative')
            plt.title('Maximum Time Derivative Map')
            plt.savefig(os.path.join(debug_output_dir, "max_time_derivative.png"))
            plt.close()
        
        # 将AIF信息保存为类属性，以便后续使用
        self.aif_position = (x, y)
        self.aif_curve = torch.tensor(aif_curve, device=self.device, dtype=torch.float32)
        
        return self.aif_curve, self.aif_position
    
    def preprocess_images(self, subtraction_images, debug_output_dir=None):
        """
        增强的图像预处理
        
        Args:
            subtraction_images: 形状为 [time_steps, height, width] 的减影图像数组
            debug_output_dir: 调试输出目录，如果不为None则保存预处理步骤的图像
            
        Returns:
            预处理后的图像张量和组织掩码
        """
        time_steps, height, width = subtraction_images.shape
        
        # 转换为张量，确保使用float32类型
        images_tensor = torch.tensor(subtraction_images, dtype=torch.float32, device=self.device)
        
        # 1. 计算时间序列的平均图像和最大图像
        mean_image = torch.mean(images_tensor, dim=0)
        max_image = torch.max(images_tensor, dim=0)[0]
        
        # 2. 创建组织掩码 - 使用Otsu阈值法
        max_image_np = max_image.cpu().numpy()
        # 归一化到0-255范围
        max_image_np_norm = (max_image_np * 255).astype(np.uint8)
        # 应用高斯模糊减少噪声
        max_image_blur = cv2.GaussianBlur(max_image_np_norm, (5, 5), 0)
        # 使用Otsu阈值法进行二值化
        _, tissue_mask_np = cv2.threshold(max_image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 应用形态学操作改进掩码
        kernel = np.ones((5, 5), np.uint8)
        # 闭运算填充小孔
        tissue_mask_np = cv2.morphologyEx(tissue_mask_np, cv2.MORPH_CLOSE, kernel)
        # 开运算去除小的孤立区域
        tissue_mask_np = cv2.morphologyEx(tissue_mask_np, cv2.MORPH_OPEN, kernel)
        
        # 4. 转换回PyTorch张量
        tissue_mask = torch.tensor(tissue_mask_np > 0, dtype=torch.bool, device=self.device)
        
        # 5. 对每个时间点应用预处理
        processed_images = []
        for t in range(time_steps):
            img_np = images_tensor[t].cpu().numpy()
            
            # 应用双边滤波保留边缘的同时减少噪声
            img_filtered = cv2.bilateralFilter(img_np, 5, 75, 75)
            
            # 只保留组织区域的信号，背景设为0
            img_masked = img_filtered * (tissue_mask_np / 255)
            
            # 对比度增强
            img_enhanced = cv2.normalize(img_masked, None, 0, 1, cv2.NORM_MINMAX)
            
            # 添加到处理后的图像列表，确保使用float32类型
            processed_images.append(torch.tensor(img_enhanced, device=self.device, dtype=torch.float32))
            
            # 保存调试图像
            if debug_output_dir is not None:
                os.makedirs(debug_output_dir, exist_ok=True)
                # 保存原始图像
                cv2.imwrite(os.path.join(debug_output_dir, f"original_t{t}.png"), 
                           (img_np * 255).astype(np.uint8))
                # 保存滤波后图像
                cv2.imwrite(os.path.join(debug_output_dir, f"filtered_t{t}.png"), 
                           (img_filtered * 255).astype(np.uint8))
                # 保存掩码后图像
                cv2.imwrite(os.path.join(debug_output_dir, f"masked_t{t}.png"), 
                           (img_masked * 255).astype(np.uint8))
                # 保存增强后图像
                cv2.imwrite(os.path.join(debug_output_dir, f"enhanced_t{t}.png"), 
                           (img_enhanced * 255).astype(np.uint8))
        
        # 保存掩码图像
        if debug_output_dir is not None:
            cv2.imwrite(os.path.join(debug_output_dir, "tissue_mask.png"), tissue_mask_np)
            cv2.imwrite(os.path.join(debug_output_dir, "max_image.png"), max_image_np_norm)
            cv2.imwrite(os.path.join(debug_output_dir, "mean_image.png"), 
                       (mean_image.cpu().numpy() * 255).astype(np.uint8))
        
        # 堆叠处理后的图像
        processed_tensor = torch.stack(processed_images)
        
        # 自动检测血管位置（AIF）- 现在调用单独的方法
        if self.aif_method == 'auto':
            self.get_auto_detected_aif(processed_tensor, tissue_mask, debug_output_dir)
        
        return processed_tensor, tissue_mask
    
    def fit_volume_gpu(self, subtraction_images, debug_output_dir=None):
        """
        使用GPU对整个体积的减影图像进行拟合
        
        Args:
            subtraction_images: 形状为 [time_steps, height, width] 的减影图像数组
            debug_output_dir: 调试输出目录
            
        Returns:
            形状为 [3, height, width] 的参数图，分别是Ktrans, ve, vp
        """
        time_steps, height, width = subtraction_images.shape
        
        print("开始预处理图像...")
        start_time = time.time()
        # 预处理图像
        images_tensor, tissue_mask = self.preprocess_images(subtraction_images, debug_output_dir)
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
                plt.xlabel('Time (min)')  # 使用英文替代
                plt.ylabel('Signal Intensity')  # 使用英文替代
                plt.title('Sample Pixel Time Curves')  # 使用英文替代
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
            
            # 打印进度
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
        
        # 后处理参数图
        print("开始后处理参数图...")
        start_time = time.time()
        
        processed_param_maps = torch.zeros_like(param_maps)
        for i in range(3):
            param_np = param_maps[i].cpu().numpy()
            
            # 应用高斯滤波平滑参数图
            param_smooth = cv2.GaussianBlur(param_np, (5, 5), 0.5)
            
            # 应用阈值去除低值
            threshold = 0.01 if i == 0 else 0.05 if i == 1 else 0.005
            param_threshold = np.where(param_smooth < threshold, 0, param_smooth)
            
            # 只保留组织区域
            param_masked = param_threshold * tissue_mask.cpu().numpy()
            
            processed_param_maps[i] = torch.tensor(param_masked, device=self.device)
            
            # 保存处理步骤的图像
            if debug_output_dir is not None:
                # 原始参数图
                plt.figure(figsize=(8, 6))
                plt.imshow(param_np, cmap='hot')
                plt.colorbar()
                plt.title(f'Original {["Ktrans", "ve", "vp"][i]} Map')  # 使用英文替代
                plt.savefig(os.path.join(debug_output_dir, f"param_{i}_original.png"))
                plt.close()
                
                # 平滑后参数图
                plt.figure(figsize=(8, 6))
                plt.imshow(param_smooth, cmap='hot')
                plt.colorbar()
                plt.title(f'Smoothed {["Ktrans", "ve", "vp"][i]} Map')  # 使用英文替代
                plt.savefig(os.path.join(debug_output_dir, f"param_{i}_smooth.png"))
                plt.close()
                
                # 阈值处理后参数图
                plt.figure(figsize=(8, 6))
                plt.imshow(param_threshold, cmap='hot')
                plt.colorbar()
                plt.title(f'Thresholded {["Ktrans", "ve", "vp"][i]} Map')  # 使用英文替代
                plt.savefig(os.path.join(debug_output_dir, f"param_{i}_threshold.png"))
                plt.close()
                
                # 最终参数图
                plt.figure(figsize=(8, 6))
                plt.imshow(param_masked, cmap='hot')
                plt.colorbar()
                plt.title(f'Final {["Ktrans", "ve", "vp"][i]} Map')  # 使用英文替代
                plt.savefig(os.path.join(debug_output_dir, f"param_{i}_final.png"))
                plt.close()
        
        print(f"后处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        return processed_param_maps.cpu().numpy()
    
    def process_patient_data(self, patient_folder, output_folder=None, debug=False):
        """
        处理单个患者的数据
        
        Args:
            patient_folder: 患者数据文件夹路径
            output_folder: 输出文件夹路径，如果为None则使用患者文件夹下的'pk_maps'子文件夹
            debug: 是否生成调试信息
            
        Returns:
            生成的参数图路径
        """
        # 设置输出文件夹
        if output_folder is None:
            output_folder = os.path.join(patient_folder, 'pk_maps')
        os.makedirs(output_folder, exist_ok=True)
        
        # 设置调试输出文件夹
        debug_output_dir = os.path.join(output_folder, 'debug') if debug else None
        if debug_output_dir is not None:
            os.makedirs(debug_output_dir, exist_ok=True)
        
        # 加载减影图像
        subtraction_images = []
        image_paths = []
        
        print(f"加载患者数据: {patient_folder}")
        for i in range(1, 9):
            sub_folder = os.path.join(patient_folder, f'SUB{i}')
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
            image_paths.append(img_path)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"警告: 无法加载图像 {img_path}")
                continue
                
            subtraction_images.append(img)
        
        if not subtraction_images:
            print(f"错误: 在 {patient_folder} 中没有找到有效的减影图像")
            return None
            
        # 打印使用的图像路径
        print("使用以下图像进行拟合:")
        for i, path in enumerate(image_paths):
            print(f"  时相 {i+1}: {path}")
            
        # 转换为numpy数组
        subtraction_images = np.array(subtraction_images)
        
        # 归一化图像
        subtraction_images = subtraction_images / 255.0
        
        # 使用GPU拟合模型
        print(f"开始为患者数据拟合Tofts模型: {patient_folder}")
        param_maps = self.fit_volume_gpu(subtraction_images, debug_output_dir)
        
        # 保存参数图
        param_names = ['ktrans', 've', 'vp']
        param_cmaps = ['hot', 'cool', 'spring']  # 使用不同的颜色映射
        saved_paths = []
        
        for i, name in enumerate(param_names):
            # 获取参数图
            param_map = param_maps[i]
            
            # 归一化参数图以便可视化
            if np.max(param_map) > 0:
                # 使用百分位数裁剪，避免极值影响
                p_min, p_max = np.percentile(param_map[param_map > 0], [1, 99])
                norm_map = np.clip(param_map, p_min, p_max)
                norm_map = ((norm_map - p_min) / (p_max - p_min) * 255).astype(np.uint8)
            else:
                norm_map = np.zeros_like(param_map, dtype=np.uint8)
                
            # 保存参数图
            output_path = os.path.join(output_folder, f'{name}.png')
            cv2.imwrite(output_path, norm_map)
            saved_paths.append(output_path)
            
            # 创建热图可视化 - 使用不同的颜色映射
            plt.figure(figsize=(8, 6))
            plt.imshow(param_map, cmap=param_cmaps[i])
            plt.colorbar(label=name)
            plt.title(f'{name.upper()} Parameter Map')
            plt.savefig(os.path.join(output_folder, f'{name}_heatmap.png'))
            plt.close()
            
            # 保存原始参数值
            np.save(os.path.join(output_folder, f'{name}_raw.npy'), param_map)
        
        # 创建RGB合成图像，将三个参数合并为一张彩色图像
        rgb_map = np.zeros((param_maps[0].shape[0], param_maps[0].shape[1], 3), dtype=np.float32)
        
        # 归一化每个参数图并分配到RGB通道
        for i in range(3):
            param = param_maps[i]
            if np.max(param) > 0:
                p_min, p_max = np.percentile(param[param > 0], [1, 99])
                norm_param = np.clip(param, p_min, p_max)
                norm_param = (norm_param - p_min) / (p_max - p_min)
            else:
                norm_param = np.zeros_like(param)
            rgb_map[:,:,i] = norm_param
        
        # 保存RGB合成图
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_map)
        plt.title('RGB Composite (R:Ktrans, G:ve, B:vp)')
        plt.savefig(os.path.join(output_folder, 'rgb_composite.png'))
        plt.close()
        
        # 保存为PNG图像
        rgb_map_8bit = (rgb_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, 'rgb_composite.png'), cv2.cvtColor(rgb_map_8bit, cv2.COLOR_RGB2BGR))
        
        return saved_paths


def test_single_patient():
    """
    测试单个患者的PK参数图拟合
    """
    # 设置数据路径
    dataset_path = '/home/wxf/project/Dataset/BreaDM'
    
    # 选择一个患者进行测试
    split = 'training'
    patient_id = None
    
    # 查找第一个有SUB序列的患者
    images_dir = os.path.join(dataset_path, 'seg', split, 'images')
    for patient in os.listdir(images_dir):
        patient_path = os.path.join(images_dir, patient)
        if not os.path.isdir(patient_path):
            continue
            
        # 检查是否有SUB序列
        has_sub = True
        for i in range(1, 9):
            if not os.path.exists(os.path.join(patient_path, f'SUB{i}')):
                has_sub = False
                break
                
        if has_sub:
            patient_id = patient
            break
    
    if patient_id is None:
        print("错误: 找不到包含完整SUB序列的患者")
        return
    
    print(f"选择患者: {patient_id}")
    
    # 设置患者路径和输出路径
    patient_path = os.path.join(images_dir, patient_id)
    output_path = os.path.join(dataset_path, 'seg', split, 'pk_maps_test', patient_id)
    
    # 创建Tofts模型拟合器
    fitter = ToftsModelFitter()
    
    # 处理患者数据，启用调试模式
    pk_maps = fitter.process_patient_data(patient_path, output_path, debug=True)
    
    if pk_maps:
        print(f"成功生成PK参数图: {pk_maps}")
    else:
        print("PK参数图生成失败")


def test_aif_methods():
    """
    测试不同AIF方法的PK参数图拟合
    """
    # 设置数据路径
    dataset_path = '/home/wxf/project/Dataset/BreaDM'
    
    # 选择一个患者进行测试
    split = 'training'
    patient_id = None
    
    # 查找第一个有SUB序列的患者
    images_dir = os.path.join(dataset_path, 'seg', split, 'images')
    for patient in os.listdir(images_dir):
        patient_path = os.path.join(images_dir, patient)
        if not os.path.isdir(patient_path):
            continue
            
        # 检查是否有SUB序列
        has_sub = True
        for i in range(1, 9):
            if not os.path.exists(os.path.join(patient_path, f'SUB{i}')):
                has_sub = False
                break
                
        if has_sub:
            patient_id = patient
            break
    
    if patient_id is None:
        print("错误: 找不到包含完整SUB序列的患者")
        return
    
    print(f"选择患者: {patient_id}")
    
    # 设置患者路径
    patient_path = os.path.join(images_dir, patient_id)
    
    # 加载减影图像
    subtraction_images = []
    image_paths = []
    
    print(f"加载患者数据: {patient_path}")
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
        image_paths.append(img_path)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"警告: 无法加载图像 {img_path}")
            continue
            
        subtraction_images.append(img)
    
    if not subtraction_images:
        print(f"错误: 在 {patient_path} 中没有找到有效的减影图像")
        return
        
    # 转换为numpy数组
    subtraction_images = np.array(subtraction_images)
    
    # 归一化图像
    subtraction_images = subtraction_images / 255.0
    
    # 测试三种AIF方法
    aif_methods = ['population', 'auto', 'modified']
    
    for method in aif_methods:
        print(f"\n测试AIF方法: {method}")
        
        # 创建输出目录
        output_path = os.path.join(dataset_path, 'seg', split, f'pk_maps_{method}', patient_id)
        os.makedirs(output_path, exist_ok=True)
        
        # 创建Tofts模型拟合器，指定AIF方法
        fitter = ToftsModelFitter(aif_method=method)
        
        # 处理患者数据，启用调试模式
        pk_maps = fitter.fit_volume_gpu(subtraction_images, debug_output_dir=os.path.join(output_path, 'debug'))
        
        # 保存参数图
        param_names = ['ktrans', 've', 'vp']
        param_cmaps = ['hot', 'cool', 'spring']  # 使用不同的颜色映射
        
        for i, name in enumerate(param_names):
            # 获取参数图
            param_map = pk_maps[i]
            
            # 归一化参数图以便可视化
            if np.max(param_map) > 0:
                # 使用百分位数裁剪，避免极值影响
                p_min, p_max = np.percentile(param_map[param_map > 0], [1, 99])
                norm_map = np.clip(param_map, p_min, p_max)
                norm_map = ((norm_map - p_min) / (p_max - p_min) * 255).astype(np.uint8)
            else:
                norm_map = np.zeros_like(param_map, dtype=np.uint8)
                
            # 保存参数图
            output_file = os.path.join(output_path, f'{name}.png')
            cv2.imwrite(output_file, norm_map)
            
            # 创建热图可视化 - 使用不同的颜色映射
            plt.figure(figsize=(8, 6))
            plt.imshow(param_map, cmap=param_cmaps[i])
            plt.colorbar(label=name)
            plt.title(f'{name.upper()} Parameter Map ({method} AIF)')
            plt.savefig(os.path.join(output_path, f'{name}_heatmap.png'))
            plt.close()
            
            # 保存原始参数值
            np.save(os.path.join(output_path, f'{name}_raw.npy'), param_map)
        
        print(f"使用 {method} AIF方法的PK参数图已保存到: {output_path}")
    
    # 比较三种方法的结果
    print("\n比较三种AIF方法的结果...")
    
    # 创建比较输出目录
    comparison_path = os.path.join(dataset_path, 'seg', split, 'pk_maps_comparison', patient_id)
    os.makedirs(comparison_path, exist_ok=True)
    
    # 加载三种方法的结果
    results = {}
    for method in aif_methods:
        method_path = os.path.join(dataset_path, 'seg', split, f'pk_maps_{method}', patient_id)
        results[method] = {}
        for param in param_names:
            param_file = os.path.join(method_path, f'{param}_raw.npy')
            if os.path.exists(param_file):
                results[method][param] = np.load(param_file)
    
    # 创建比较图
    for param in param_names:
        plt.figure(figsize=(15, 5))
        
        for i, method in enumerate(aif_methods):
            if param in results[method]:
                plt.subplot(1, 3, i+1)
                plt.imshow(results[method][param], cmap=param_cmaps[param_names.index(param)])
                plt.colorbar(label=param)
                plt.title(f'{param.upper()} ({method} AIF)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_path, f'{param}_comparison.png'))
        plt.close()
    
    # 计算方法间的差异
    for param in param_names:
        plt.figure(figsize=(15, 5))
        
        # 计算相对差异
        for i, (method1, method2) in enumerate([('population', 'auto'), ('population', 'modified'), ('auto', 'modified')]):
            if param in results[method1] and param in results[method2]:
                diff = results[method1][param] - results[method2][param]
                
                plt.subplot(1, 3, i+1)
                plt.imshow(diff, cmap='bwr')
                plt.colorbar(label='Difference')
                plt.title(f'{param.upper()}: {method1} vs {method2}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_path, f'{param}_difference.png'))
        plt.close()
    
    print(f"比较结果已保存到: {comparison_path}")


if __name__ == "__main__":
    # test_single_patient()
    test_aif_methods()


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
    # 注意：在实际应用中，可能需要根据扫描参数和对比剂特性进行更复杂的转换
    # 例如，对于T1加权图像，可以使用 C = (1/r1) * (1/T1 - 1/T10)
    # 其中r1是对比剂的松弛率，T1和T10分别是增强后和增强前的T1值
    concentration = relative_enhancement
    
    return concentration