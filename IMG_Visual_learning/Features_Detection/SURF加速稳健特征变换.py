#SURF加速稳健特征：
#   一种基于 SIFT 的改进算法：
# 旨在提供更快的计算速度和更好的性能。
# 
# SURF通过使用积分图像和Haar小波响应来加速特征检测和描述符计算过程。

# 它在保证检测效果基本不变的前提下，通过优化将运行速度提升好几倍。
# 适合用在视频目标跟踪、手机端图像匹配、实时全景拼接等对速度有要求的场景。
# 实现效果与效率的兼顾.

#1.基础优化：
# 1. 使用积分图像：
#   SURF 使用积分图像来加速特征检测过程
#   积分图像允许快速计算图像区域的和
#   从而加快了特征点的检测速度
# 核心优势：
# 任意矩形区域的像素和可通过3次加法 （A-B-C+D）计算
# 与区域大小无关，将卷积运算复杂度从O(n^2)降至O(1)，
# 为后续快速滤波提供基础。

# 2. 特征检测：
# 利用Hessian矩阵的行列式和迹来检测特征点
# SURF 通过盒式滤波器（Box Filter）近似方法替代了高斯二阶导数。

# 3. 特征描述符：
# SURF 使用 Haar 小波响应来描述特征点的局部特征，这比 SIFT 的梯度直方图描述符更简单，计算速度更快。

# 4.特征匹配：
# SURF 生成的特征描述符是基于 Haar 小波响应的，具有较好的区分能力和旋转不变性。
# 这使得 SURF 在特征匹配过程中能够更准确地识别图像中的相似区域。

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#SURF算法由于专利未到期，头文件不存在于cv2中，无法直接使用。
# 需要安装opencv-contrib-python库才能使用SURF算法
#或者使用SIFT的方法进行修改就可以使用SURF算法了。
#
# 
# 以下是具体的修改方案✨：

def surf_sift_matching(img1_path,img2_path):
    # 1. 读取图像
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("无法读取图像，请检查路径是否正确。")
        print(f"img1_path: {img1_path}, img2_path: {img2_path}")
        return
    
    # 2. 创建SIFT检测器对象(替代SURF)(接口是一样的，以后可以换成SURF)
    sift = cv.SIFT_create(  
        nfeatures=0,          # 检测的最大特征点数（0=无限制）
        nOctaveLayers=3,      # 每个八度的尺度层数（与SURF一致）
        contrastThreshold=0.04,# 对比度阈值（替代SURF的Hessian阈值，值越小特征点越多）
        edgeThreshold=10,     # 边缘阈值（过滤边缘特征点）
        sigma=1.6  
    )

    # 3. 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"图像1检测到的关键点数: {len(kp1)}, 描述符维度: {des1.shape if des1 is not None else 'None'}")
    print(f"图像2检测到的关键点数: {len(kp2)}, 描述符维度: {des2.shape if des2 is not None else 'None'}")

    # 4. 绘制关键点：
    img1_kp = cv.drawKeypoints(img1,kp1,des1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv.drawKeypoints(img2,kp2,des2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # 5. 特征匹配：
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 指定检查次数
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 6. 简单的NNDR匹配过滤
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"通过NNDR过滤后的匹配数: {len(good_matches)}")


    # 绘制匹配结果
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    # 显示匹配结果
    plt.figure(figsize=(15, 8))
    plt.subplot(1,3,1, title=f'PPM图像1关键点（共{len(kp1)}个）')
    plt.imshow(cv.cvtColor(img1_kp, cv.COLOR_BGR2RGB))
    plt.subplot(1,3,2, title=f'PPM图像2关键点（共{len(kp2)}个）')
    plt.imshow(cv.cvtColor(img2_kp, cv.COLOR_BGR2RGB))
    plt.subplot(1,3,3, title=f'SIFT 特征匹配（有效匹配{len(good_matches)}个）')
    plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


surf_sift_matching('Features_Detection\\BB1.jpg', 'Features_Detection\\BB2.jpg')













