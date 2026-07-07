# ============================================================
# ORB + SIFT 实现图像拼接（Image Stitching）
# ============================================================
# 整体流程：
#   1. 读取两张有重叠区域的图片
#   2. 分别用 ORB 和 SIFT 提取特征点和描述子
#   3. 用暴力匹配器 + Lowe's ratio test 筛选匹配点
#   4. 用 SIFT 的匹配点 + RANSAC 估计单应矩阵（Homography）
#   5. 对左图做透视变换，拼接到右图上
#   6. 输出全景拼接图
# ============================================================

import cv2 as cv
import numpy as np
from pathlib import Path


# ---------- 辅助函数 ----------
def img_show(name, img, max_width=1200):
    """
    显示图像，自动缩放到屏幕可显示的大小
    参数：
        name: 窗口名称
        img:  要显示的图像
        max_width: 窗口最大宽度（像素），超过则等比缩放
    """
    h, w = img.shape[:2]
    if w > max_width:
        # scale = 目标宽度 / 原始宽度，保持宽高比
        scale = max_width / w
        img = cv.resize(img, None, fx=scale, fy=scale)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# ---------- 1. 读取图像 ----------
base_dir = Path(__file__).resolve().parent

img_1_path = base_dir / 'Stitching_img1.jpg'  # 左图（将被透视变换）
img_2_path = base_dir / 'Stitching_img2.jpg'  # 右图（作为基准）

img_1 = cv.imread(str(img_1_path))
img_2 = cv.imread(str(img_2_path))
assert img_1 is not None, f"图像加载失败: {img_1_path}"
assert img_2 is not None, f"图像加载失败: {img_2_path}"

print(f"左图尺寸: {img_1.shape}")
print(f"右图尺寸: {img_2.shape}")

img_show('左图', img_1)
img_show('右图', img_2)


# ---------- 2. SIFT 特征提取 ----------
# SIFT：在不同尺度空间检测关键点，计算 128 维浮点描述子
# 优点：对旋转、尺度、光照变化鲁棒
sift = cv.SIFT_create(
    nfeatures=5000,          # 最多检测的关键点数量
    nOctaveLayers=3,         # 每个八度的尺度层数
    contrastThreshold=0.04,  # 对比度阈值（越小检测越多）
    edgeThreshold=10,        # 边缘阈值
    sigma=1.6                # 高斯核初始标准差
)

# detectAndCompute(image, mask) → (keypoints, descriptors)
#   keypoints: 关键点列表，每个包含坐标 (pt)、尺度、角度
#   descriptors: 描述子数组，形状 (N, 128)
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)
print(f"SIFT: 左图 {len(kp1)} 个关键点，右图 {len(kp2)} 个")


# ---------- 3. ORB 特征提取（作为对比） ----------
# ORB：FAST 角点 + BRIEF 二进制描述子，速度快
orb = cv.ORB_create(nfeatures=2000)

# ORB descriptors 是二进制描述子，形状 (N, 32)
kp1_orb, des1_orb = orb.detectAndCompute(img_1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img_2, None)
print(f"ORB:  左图 {len(kp1_orb)} 个关键点，右图 {len(kp2_orb)} 个")


# ---------- 4. 特征匹配 ----------
# BFMatcher：暴力匹配，对每个描述子遍历所有描述子计算距离
#   NORM_HAMMING: 汉明距离，适用于二进制描述子（ORB）
#   NORM_L2:      欧氏距离，适用于浮点描述子（SIFT）
bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
bf_sift = cv.BFMatcher(cv.NORM_L2, crossCheck=False)


# ---------- 5. Lowe's Ratio Test ----------
# knnMatch(des1, des2, k=2): 对每个描述子找 2 个最近邻
# ratio < 0.75: 最近匹配显著优于次近匹配 → 可靠匹配
matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
good_orb = [m for m, n in matches_orb if m.distance < 0.75 * n.distance]

matches_sift = bf_sift.knnMatch(des1, des2, k=2)
good_sift = [m for m, n in matches_sift if m.distance < 0.75 * n.distance]

print(f"ORB  优质匹配: {len(good_orb)}")
print(f"SIFT 优质匹配: {len(good_sift)}")


# ---------- 6. 估计单应矩阵 ----------
# findHomography 用 RANSAC 估计 3×3 单应矩阵 H
# H 描述左图到右图的透视变换关系
# 要求至少 4 对匹配点
if len(good_sift) < 4:
    print(f"SIFT 匹配点不足（{len(good_sift)} < 4），无法拼接")
    exit()

# 提取匹配点坐标，形状 (N, 1, 2)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

# RANSAC: ransacReprojThreshold=5.0 表示重投影误差阈值 5 像素
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

print(f"SIFT 内点数: {int(sum(matches_mask))} / {len(good_sift)}")


# ---------- 7. 透视变换并拼接 ----------
h1, w1 = img_1.shape[:2]  # 左图 (高, 宽)
h2, w2 = img_2.shape[:2]  # 右图 (高, 宽)

# 7a. 计算左图四个角点变换后的位置
corners_img1 = np.float32([
    [0, 0], [0, h1], [w1, h1], [w1, 0]
]).reshape(-1, 1, 2)
warped_corners = cv.perspectiveTransform(corners_img1, H)

# 7b. 计算画布范围（合并变换后的左图角点 + 右图角点）
corners_img2 = np.float32([
    [0, 0], [0, h2], [w2, h2], [w2, 0]
]).reshape(-1, 1, 2)

all_corners = np.concatenate((warped_corners, corners_img2), axis=0)

# 找最小/最大坐标
all_pts = all_corners.reshape(-1, 2)
xmin, ymin = np.int32(all_pts.min(axis=0))
xmax, ymax = np.int32(all_pts.max(axis=0))

# 7c. 平移矩阵：把所有坐标移到正数区域
# xmin 可能是负数（左图变换后可能跑到左边外面）
tx = -xmin  # 水平平移量
ty = -ymin  # 垂直平移量

T = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
], dtype=np.float64)

# 最终变换矩阵 = 平移 × 单应矩阵（先透视变换，再平移）
final_H = T @ H

# 7d. 执行透视变换
canvas_w = xmax - xmin
canvas_h = ymax - ymin
stitched = cv.warpPerspective(img_1, final_H, (canvas_w, canvas_h))

# 7e. 将右图放置到画布上
# 右图在画布上的位置：x 方向偏移 tx，y 方向偏移 ty
stitched[ty:ty+h2, tx:tx+w2] = img_2

print(f"拼接完成！输出尺寸: {stitched.shape[1]}x{stitched.shape[0]}")


# ---------- 8. 可视化 ----------
# 画匹配图（只显示内点）
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=2
)
match_img = cv.drawMatches(img_1, kp1, img_2, kp2, good_sift, None, **draw_params)

img_show('SIFT 匹配（绿线=内点）', match_img)
img_show('拼接结果', stitched)

# 保存结果
output_path = base_dir / 'stitching_result.jpg'
cv.imwrite(str(output_path), stitched)
print(f"结果已保存到: {output_path}")
