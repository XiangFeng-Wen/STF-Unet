import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualConvBlock(nn.Module):
    input_format = "time_sequence"  # 输入为 [B, T, C, H, W]
    """双层残差卷积模块，用于提取局部空间上下文信息"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """解码器块，用于上采样和特征融合"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 转置卷积上采样 (3x3, 步长=2)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 特征融合后的通道压缩 (1x1卷积)
        self.fusion = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        
        # 双层残差卷积模块
        self.res_conv = ResidualConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        # 上采样
        x = self.up(x)
        
        # 确保尺寸匹配
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        # 特征融合（通道维拼接）
        x = torch.cat([x, skip], dim=1)
        
        # 通道压缩 (1x1卷积)
        x = self.fusion(x)
        
        # 双层残差卷积
        x = self.res_conv(x)
        
        return x


class TimeDistributed(nn.Module):
    """将任何层应用于输入的每个时间步"""
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
    
    def forward(self, x):
        # x shape: [batch, time_steps, channels, height, width]
        batch_size, time_steps, C, H, W = x.size()
        # 重塑为 [batch*time_steps, channels, height, width]
        x_reshaped = x.contiguous().view(batch_size * time_steps, C, H, W)
        # 应用模块
        y = self.module(x_reshaped)
        # 重塑回 [batch, time_steps, ...]
        y_reshaped = y.contiguous().view(batch_size, time_steps, *y.size()[1:])
        return y_reshaped


class STFLSTMUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, time_steps=8, use_pk_maps=False, pk_channels=3):
        super(STFLSTMUNet, self).__init__()
        self.time_steps = time_steps
        self.use_pk_maps = use_pk_maps
        self.pk_channels = pk_channels if use_pk_maps else 0
        
        # 计算实际输入通道数 - 如果使用PK特征图，则需要增加通道数
        actual_in_channels = in_channels
        if use_pk_maps:
            actual_in_channels += pk_channels
        
        # 加载ResNet-34作为编码器，不使用预训练权重
        resnet = models.resnet34(weights=None)
        
        # 修改第一层以接受正确的输入通道数
        self.conv1 = nn.Conv2d(actual_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 编码器阶段
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 输出通道: 64, 尺寸: 1/4
        self.layer2 = resnet.layer2  # 输出通道: 128, 尺寸: 1/8
        self.layer3 = resnet.layer3  # 输出通道: 256, 尺寸: 1/16
        self.layer4 = resnet.layer4  # 输出通道: 512, 尺寸: 1/32
        
        # PK参数融合层 - 在LSTM之前融合
        if use_pk_maps:
            self.pk_fusion1 = nn.Conv2d(64 + pk_channels, 64, kernel_size=1)
            self.pk_fusion2 = nn.Conv2d(128 + pk_channels, 128, kernel_size=1)
            self.pk_fusion3 = nn.Conv2d(256 + pk_channels, 256, kernel_size=1)
            self.pk_fusion4 = nn.Conv2d(512 + pk_channels, 512, kernel_size=1)
        
        # LSTM模块 - 处理时序特征
        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.lstm3 = nn.LSTM(256, 256, batch_first=True)
        self.lstm4 = nn.LSTM(512, 512, batch_first=True)
        
        # 解码器阶段 - 更新为新的解码器块
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        
        # 最终上采样和分类
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_res = ResidualConvBlock(32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x, pk_maps=None):
        # x的形状为 [batch_size, time_steps+pk_channels, channels, height, width]
        # 当use_pk_maps=True时，PK特征图已经附加在x的时间步维度的末尾
        
        batch_size, total_steps, channels, height, width = x.shape
        
        # 如果启用了PK特征图，从输入中提取PK特征图
        if self.use_pk_maps:
            # 分离时间序列和PK特征图
            time_steps = total_steps - self.pk_channels
            time_series = x[:, :time_steps, :, :, :]
            extracted_pk_maps = x[:, time_steps:, :, :, :]
            # 调整PK特征图的形状为 [batch_size, pk_channels, height, width]
            extracted_pk_maps = extracted_pk_maps.reshape(batch_size, self.pk_channels, channels, height, width)
            extracted_pk_maps = extracted_pk_maps.squeeze(2)  # 移除channels维度，假设channels=1
            x = time_series
            # 使用提取的PK特征图，忽略传入的pk_maps参数
            pk_maps = extracted_pk_maps
        else:
            # 如果不使用PK特征图，则时间步就是总步数
            time_steps = total_steps
            pk_maps = None
        
        # 编码阶段 - 处理每个时间步
        enc1_features = []
        enc2_features = []
        enc3_features = []
        enc4_features = []
        
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # [batch_size, channels, height, width]
            
            # 如果使用PK特征图，在输入阶段就拼接PK特征
            if self.use_pk_maps and pk_maps is not None:
                # 拼接PK特征图到输入
                x_t = torch.cat([x_t, pk_maps], dim=1)
            
            # 第一阶段
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t = self.relu(x_t)
            e1 = self.maxpool(x_t)  # 1/4
            
            # 后续阶段
            e1 = self.layer1(e1)    # 1/4
            e2 = self.layer2(e1)    # 1/8
            e3 = self.layer3(e2)    # 1/16
            e4 = self.layer4(e3)    # 1/32
            
            # 如果使用PK参数，在这里融合PK特征 (可选的额外融合，可以根据需要保留或移除)
            if self.use_pk_maps and pk_maps is not None and hasattr(self, 'pk_fusion1'):
                # 调整PK图谱尺寸以匹配各个特征图
                pk_maps_1 = F.interpolate(pk_maps, size=e1.shape[2:], mode='bilinear', align_corners=True)
                pk_maps_2 = F.interpolate(pk_maps, size=e2.shape[2:], mode='bilinear', align_corners=True)
                pk_maps_3 = F.interpolate(pk_maps, size=e3.shape[2:], mode='bilinear', align_corners=True)
                pk_maps_4 = F.interpolate(pk_maps, size=e4.shape[2:], mode='bilinear', align_corners=True)
                
                # 融合PK参数到残差卷积输出
                e1 = self.pk_fusion1(torch.cat([e1, pk_maps_1], dim=1))
                e2 = self.pk_fusion2(torch.cat([e2, pk_maps_2], dim=1))
                e3 = self.pk_fusion3(torch.cat([e3, pk_maps_3], dim=1))
                e4 = self.pk_fusion4(torch.cat([e4, pk_maps_4], dim=1))
            
            # 存储融合后的特征
            enc1_features.append(e1)
            enc2_features.append(e2)
            enc3_features.append(e3)
            enc4_features.append(e4)
        
        # 将特征堆叠为时序数据
        enc1_seq = torch.stack(enc1_features, dim=1)  # [batch_size, time_steps, 64, h/4, w/4]
        enc2_seq = torch.stack(enc2_features, dim=1)  # [batch_size, time_steps, 128, h/8, w/8]
        enc3_seq = torch.stack(enc3_features, dim=1)  # [batch_size, time_steps, 256, h/16, w/16]
        enc4_seq = torch.stack(enc4_features, dim=1)  # [batch_size, time_steps, 512, h/32, w/32]
        
        # 应用LSTM处理时序特征 - 使用ConvLSTM方式处理每个像素位置的时序特征
        # 对于每个特征图，我们分别处理每个空间位置的时序信息
        b, t, c, h, w = enc1_seq.shape
        # 重塑为 [batch_size*height*width, time_steps, channels]
        enc1_lstm_in = enc1_seq.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        enc1_lstm_out, _ = self.lstm1(enc1_lstm_in)
        # 重塑回 [batch_size, time_steps, channels, height, width]
        enc1_lstm_out = enc1_lstm_out.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        
        b, t, c, h, w = enc2_seq.shape
        enc2_lstm_in = enc2_seq.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        enc2_lstm_out, _ = self.lstm2(enc2_lstm_in)
        enc2_lstm_out = enc2_lstm_out.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        
        b, t, c, h, w = enc3_seq.shape
        enc3_lstm_in = enc3_seq.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        enc3_lstm_out, _ = self.lstm3(enc3_lstm_in)
        enc3_lstm_out = enc3_lstm_out.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        
        b, t, c, h, w = enc4_seq.shape
        enc4_lstm_in = enc4_seq.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        enc4_lstm_out, _ = self.lstm4(enc4_lstm_in)
        enc4_lstm_out = enc4_lstm_out.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        
        # 使用最后一个时间步的LSTM输出作为解码器输入
        enc1_feat = enc1_lstm_out[:, -1, :, :, :]  # [batch_size, 64, h/4, w/4]
        enc2_feat = enc2_lstm_out[:, -1, :, :, :]  # [batch_size, 128, h/8, w/8]
        enc3_feat = enc3_lstm_out[:, -1, :, :, :]  # [batch_size, 256, h/16, w/16]
        enc4_feat = enc4_lstm_out[:, -1, :, :, :]  # [batch_size, 512, h/32, w/32]
        
        # 解码阶段 - 使用LSTM增强的特征和跳跃连接
        dec4 = self.decoder4(enc4_feat, enc3_feat)  # 上采样并融合enc3特征
        dec3 = self.decoder3(dec4, enc2_feat)       # 上采样并融合enc2特征
        dec2 = self.decoder2(dec3, enc1_feat)       # 上采样并融合enc1特征
        
        # 最后一次上采样回到原始分辨率
        dec1 = self.upconv1(dec2)
        dec1 = self.final_res(dec1)
        
        # 最终分类
        output = self.final(dec1)
        
        return {'out': output}


# 辅助函数：提取PK参数图谱
def extract_pk_maps(x):
    """
    从输入张量中提取PK参数图谱
    
    Args:
        x: 输入张量，形状为 [batch_size, time_steps+pk_channels, channels, height, width]
           其中最后3个时间步是PK参数图谱
    
    Returns:
        time_series: 时间序列部分，形状为 [batch_size, time_steps, channels, height, width]
        pk_maps: PK参数图谱，形状为 [batch_size, pk_channels, height, width]
    """
    batch_size, total_steps, channels, height, width = x.shape
    time_steps = total_steps - 3  # 减去3个PK参数通道
    
    time_series = x[:, :time_steps, :, :, :]
    pk_maps = x[:, time_steps:, :, :, :]
    
    # 调整PK参数图谱的形状
    pk_maps = pk_maps.reshape(batch_size, 3, channels, height, width)
    pk_maps = pk_maps.squeeze(2)  # 移除channels维度，假设channels=1
    
    return time_series, pk_maps