#----------------ORB（Oriented FAST and Rotated BRIEF）----------------
# ORB是一种结合了FAST角点检测和BRIEF描述子的特征检测和描述算法。它具有旋转不变性和尺度不变性，适用于实时应用。
# ORB的主要步骤包括：   
'''
这是一个在计算成本、性能匹配以及主要是专利方面，
                是SIFT和SURF的良好替代方案。 
是的，SIFT和SURF是有专利的，你应该付费使用它们。
但ORB并不!!
🐂🍺
'''
# ORB基本上是FAST关键点检测器和BRIEF描述符的融合，并提升性能.
# 但有一个问题是FAST不计算方向
# 那么旋转不变性呢？----利用Rotated BRIEF描述符来实现旋转不变性。
# ORB通过计算关键点的主方向来实现旋转不变性。

# 额外说明：ORB的旋转不变性来自两部分：
# 1) 关键点方向（orientation）：ORB 使用图像块的强度质心（intensity centroid）来估计每个关键点的主方向。
#    具体做法是计算关键点邻域内像素的第一矩（基于像素坐标和强度），主方向由质心相对于关键点的角度给出。
# 2) 旋转后的描述子（Rotated BRIEF）：得到关键点方向后，ORB 会按该方向旋转 BRIEF 的采样模式，
#    再在旋转后的采样位置上计算二值描述子。这使得当图像或关键点旋转时，描述子保持一致，因而具有旋转不变性。
#
# 组合起来：关键点方向保证了描述子采样模式与图像局部旋转对齐，旋转后的 BRIEF 描述子因此对图像旋转具有鲁棒性。

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('IMG_Visual_learning\\Features_Detection\\BB1.jpg')
assert img is not None, f"图像加载失败，请检查路径{img}是否正确。"
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

orb_create = getattr(cv, 'ORB_create')

#初始化ORB检测器对象：
orb = orb_create(
    nfeatures=1000,
    scoreType=cv.ORB_HARRIS_SCORE
    )  # 设置最大特征数量和评分类型

#检测关键点：
kp = orb.detect(gray,None)
#计算描述符:
kp, des = orb.compute(gray, kp)

## draw only keypoints location,not size and orientation:
img2 = cv.drawKeypoints(img, kp, img.copy(), color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
# 显示结果  
plt.imshow(img2)
plt.title('ORB Keypoints')
plt.axis('off')
plt.show()

# 对BB1和BB2进行ORB特征匹配，使用FLANN匹配器
img1 = cv.imread('IMG_Visual_learning\\Features_Detection\\BB1.jpg')
img2 = cv.imread('IMG_Visual_learning\\Features_Detection\\BB2.jpg')
assert img1 is not None, "图像加载失败，请检查BB1路径是否正确。"
assert img2 is not None, "图像加载失败，请检查BB2路径是否正确。"

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create(nfeatures=1000, scoreType=cv.ORB_HARRIS_SCORE)
orb = orb_create(nfeatures=1000, scoreType=cv.ORB_HARRIS_SCORE)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

img1_draw = cv.drawKeypoints(img1, kp1, img1.copy(), color=(0, 255, 0), flags=cv.DrawMatchesFlags_DEFAULT)
img2_draw = cv.drawKeypoints(img2, kp2, img2.copy(), color=(0, 255, 0), flags=cv.DrawMatchesFlags_DEFAULT)
# FLANN参数：ORB为二进制描述子，使用LSH索引
FLANN_INDEX_LSH = 6
index_params: dict[str, int] = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1,
)
search_params: dict[str, int] = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2) if des1 is not None and des2 is not None else []

# Lowe比率测试筛选匹配点
good_matches = []
for pair in matches:
    if len(pair) < 2:
        continue
    m, n = pair
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

match_img = cv.drawMatches(
    img1, kp1, img2, kp2, good_matches, img1.copy(),
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 统计关键点和有效匹配数
kp1_count = len(kp1)
kp2_count = len(kp2)
good_matches_count = len(good_matches)

# 顶部两个子图分别显示两张原图及各自关键点数，底部显示匹配结果并在下方标注有效匹配数
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.imshow(cv.cvtColor(img1_draw, cv.COLOR_BGR2RGB))
plt.title(f'Image 1 - Keypoints: {kp1_count}')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(cv.cvtColor(img2_draw, cv.COLOR_BGR2RGB))
plt.title(f'Image 2 - Keypoints: {kp2_count}')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
plt.title(f'ORB + FLANN Feature Matching - Good matches: {good_matches_count}')
plt.axis('off')

plt.tight_layout()
plt.show()











