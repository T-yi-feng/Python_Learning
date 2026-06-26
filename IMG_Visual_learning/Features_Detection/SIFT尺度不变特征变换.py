# SIFT（尺度不变特征变换）:
# SIFT 是一种用于检测和描述局部特征的算法:
#       具有尺度不变性和旋转不变性。
# 它通过在不同尺度上检测关键点，并为每个关键点计算一个描述符来实现特征匹配。
# ✨SIFT 的主要步骤包括：
# 1. 尺度空间极值检测：在不同尺度上检测关键点。
# 2. 关键点定位：精确定位关键点位置。
#
# 函数：cv.SIFT_create()

'''
理论：
1. 尺度空间极值检测：
   - SIFT 通过构建尺度空间来检测关键点。
          尺度空间是通过对图像进行高斯模糊来创建的。
   - 在尺度空间中，SIFT 通过比较不同尺度上的图像来检测关键点。
          具体来说，SIFT 计算图像在不同尺度上的差分（DoG）图像。
          并在这些图像中寻找局部极值点。

2. 关键点定位：在检测到关键点后，进一步精确定位关键点的位置和尺度.
            这里利用泰勒级数展开尺度空间来获得更准确的极值位置，
            并通过设置contrastThreshold和edgeThreshold来过滤掉不稳定的关键点。

3. 方向分配：为每个关键点分配一个主方向，以实现旋转不变性。

4. 关键点描述符：为每个关键点计算一个描述符，通常是一个128维的向量。
          该描述符基于关键点周围的梯度信息构建，具有旋转和尺度不变性。

5. 特征匹配：使用描述符进行特征匹配，通常使用欧氏距离或其他距离度量来比较描述符之间的相似性。
          SIFT 的描述符具有较高的区分能力，使得它在图像匹配和物体识别等任务中表现出色。

'''

#------------OpenCV-Python SIFT 示例代码--------------#
import cv2 as cv
import numpy as np
import os
from PIL import Image
import matplotlib
# 设置matplotlib使用非交互式后端，避免Qt依赖问题
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def imread_chinese_path(path):
    """读取包含中文路径的图片"""
    try:
        img = Image.open(path)
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
            return img_bgr
        return img_np
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None

def imwrite_chinese_path(path, img):
    """保存图片到中文路径"""
    try:
        # 如果是BGR格式，转换为RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img_rgb = img
        # 使用PIL保存
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(path)
        print(f"图片已保存到: {path}")
        return True
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取图像 - 使用绝对路径和自定义函数
img_path = os.path.join(script_dir, 'chorch.jpg')
img = imread_chinese_path(img_path)
assert img is not None, f"图像读取失败，请检查路径是否正确: {img_path}"
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#1. 创建 SIFT 检测器对象：
# 创建一个 SIFT 检测器对象（调用 OpenCV 的 SIFT 构造函数）
# 语法: cv.SIFT_create() -> 返回一个包含 SIFT 方法的对象（例如 detect, compute）
sift = cv.SIFT_create(  
    nfeatures=5000,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6
)

#2.关键点定位：
# 使用 sift 对象检测关键点（keypoints）
# 语法: detector.detect(image, mask=None) -> 返回关键点列表
# 这里传入 gray（灰度图），第二个参数 None 表示不使用掩码
kp = sift.detect(gray, None)

#3.描述关键点：
img_with_kp = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 保存结果图片 - 使用中文路径兼容函数
output_path = os.path.join(script_dir, 'chorch_sift_keypoints.jpg')
imwrite_chinese_path(output_path, img_with_kp)

# 显示结果 - 使用matplotlib保存为文件而不是显示窗口
plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')

#4. 现在获取了关键点描述符，可以计算这些描述符来进行特征匹配等后续处理。
# 
# 计算关键点的描述符：
# 语法: detector.compute(image, keypoints) 
#               -> 返回关键点列表和对应的描述符数组
#eg：sift.compute(gray, kp) -> 返回 kp（关键点列表）和 des（描述符数组）
kp, des = sift.compute(gray, kp)
print(f"检测到 {len(kp)} 个关键点，描述符维度: {des.shape}")

#result:检测到 12494 个关键点，描述符维度: (12494, 128)
#       这里 kp 是关键字点列表，des 是一个形状的数组(kp)*128.


#缺点：
# 1. 计算复杂度高：✨
#   SIFT 的计算过程较为复杂，尤其是在检测和描述符计算阶段，可能会导致处理速度较慢。
# 2. 专利限制：SIFT 曾经受到专利保护，虽然现在已经过期，但在某些应用中可能仍然存在法律风险。
# 3. 对于某些类型的图像（如纹理较少的图像），SIFT 可能无法检测到足够的关键点，导致匹配效果不佳。

#故：SURF（加速稳健特征）和 ORB（Oriented FAST and Rotated BRIEF）等算法、
# 被开发出来，以提供更快的计算速度和更好的性能
#               同时避免专利问题    



















































