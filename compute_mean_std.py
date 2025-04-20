import os
from PIL import Image
import numpy as np
import cv2


def main():
    # 设置为灰度图像（单通道）
    img_channels = 1
    # BreaDM数据集路径
    data_root = "/home/wxf/project/Dataset/BreaDM"
    img_dir = os.path.join(data_root, "seg", "training", "images")
    
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    # 获取所有图像文件
    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg") or i.endswith(".png")]
    cumulative_mean = 0.0
    cumulative_std = 0.0
    valid_images = 0
    
    print(f"Processing {len(img_name_list)} images...")
    
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        try:
            # 读取图像为灰度图
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # 归一化到[0,1]范围
            img = img.astype(np.float32) / 255.0
            
            # 累加均值和标准差
            cumulative_mean += np.mean(img)
            cumulative_std += np.std(img)
            valid_images += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if valid_images == 0:
        print("No valid images found!")
        return
        
    # 计算平均值
    mean = cumulative_mean / valid_images
    std = cumulative_std / valid_images
    
    print(f"Processed {valid_images} valid images")
    print(f"mean: {mean:.3f}")
    print(f"std: {std:.3f}")


if __name__ == '__main__':
    main()