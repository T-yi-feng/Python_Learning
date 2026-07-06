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


# ---------- 1. 读取图像 ----------
# 获取脚本所在目录，用 Path 处理路径，兼容 Windows/Mac/Linux
base_dir = Path(__file__).resolve().parent

# 拼接两张图片的路径（左边的图和右边的图，它们有重叠区域）
img_1_path = base_dir / 'Stitching_img1.jpg'  # 左图（将被透视变换）
img_2_path = base_dir / 'Stitching_img2.jpg'  # 右图（作为基准）

# cv.imread(filename, flags) 读取图像
#   flags: cv.IMREAD_COLOR (默认, BGR 三通道) / cv.IMREAD_GRAYSCALE (灰度)
#   返回值: numpy.ndarray，形状为 (H, W, 3) 或 (H, W)
img_1 = cv.imread(str(img_1_path))
img_2 = cv.imread(str(img_2_path))

# assert 断言：如果条件为 False，抛出异常并打印后面的字符串
# 用途：确保图像加载成功，避免后续代码在空图上运行
assert img_1 is not None, f"图像加载失败，请检查路径是否正确：{img_1_path}"
assert img_2 is not None, f"图像加载失败，请检查路径是否正确：{img_2_path}"

print(f"左图尺寸: {img_1.shape}")  # 输出 (高度, 宽度, 通道数)
print(f"右图尺寸: {img_2.shape}")


# ---------- 辅助函数：显示图像 ----------
def img_show(name, img):
    """
    显示图像并等待按键关闭窗口
    参数：
        name: 窗口名称（字符串）
        img:  要显示的图像（numpy 数组）
    """
    cv.imshow(name, img)       # 创建/更新窗口并显示图像
    cv.waitKey(0)              # 等待任意按键（0 = 无限等待）
    cv.destroyAllWindows()     # 关闭所有 OpenCV 窗口


# 显示原始图像，确认读取正确
img_show('左图 (Image 1)', img_1)
img_show('右图 (Image 2)', img_2)


# ---------- 2. SIFT 特征提取 ----------
# SIFT（尺度不变特征变换）：在不同尺度空间检测关键点，并计算 128 维浮点描述子
# 优点：对旋转、尺度、光照变化鲁棒
# 缺点：速度较慢（相比 ORB）
sift = cv.SIFT_create(
    nfeatures=5000,          # 最多检测的关键点数量（越多越精细，但越慢）
    nOctaveLayers=3,         # 每个八度的尺度层数（3 是标准值，影响尺度分辨率）
    contrastThreshold=0.04,  # 对比度阈值（越小检测到的点越多，包括低对比度区域）
    edgeThreshold=10,        # 边缘阈值（过滤掉边缘响应点，越大保留越多边缘点）
    sigma=1.6                # 高斯核的初始标准差（影响最粗尺度的模糊程度）
)

# detectAndCompute(image, mask) 一次性完成检测和计算
#   image: 输入图像（灰度或彩色，SIFT 内部会转灰度）
#   mask:  掩码，指定在哪些区域检测（None = 全图）
#   返回值: (keypoints, descriptors)
#     keypoints: 关键点列表，每个包含坐标 (pt)、尺度 (size)、角度 (angle) 等
#     descriptors: 描述子数组，形状 (N, 128)，N 是关键点数量，每个关键点 128 维向量
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)
print(f"SIFT: 左图检测到 {len(kp1)} 个关键点，右图检测到 {len(kp2)} 个")


# ---------- 3. ORB 特征提取（作为对比） ----------
# ORB（Oriented FAST and Rotated BRIEF）：FAST 角点 + BRIEF 二进制描述子
# 优点：速度快（比 SIFT 快 100 倍），免费无专利
# 缺点：对尺度和旋转的鲁棒性不如 SIFT
orb = cv.ORB_create(
    nfeatures=2000            # 最多检测的关键点数量
    # ORB 的其他默认参数已经足够好，一般不需要调整
)

# detectAndCompute 返回 keypoints 和 descriptors
# ORB 的 descriptors 是二进制描述子，形状 (N, 32)，每个描述子 256 位（32 字节）
kp1_orb, des1_orb = orb.detectAndCompute(img_1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img_2, None)
print(f"ORB:  左图检测到 {len(kp1_orb)} 个关键点，右图检测到 {len(kp2_orb)} 个")


# ---------- 4. 特征匹配 ----------
# BFMatcher（暴力匹配器）：对每个描述子，遍历所有描述子计算距离，取最近的
#   normType: 距离度量方式
#     cv.NORM_L1    → L1 距离（绝对值差之和），适用于 float 描述子（如 SIFT）
#     cv.NORM_L2    → L2 距离（欧氏距离），适用于 float 描述子
#     cv.NORM_HAMMING → 汉明距离（不同位的个数），适用于二进制描述子（如 ORB）
#   crossCheck: 是否启用交叉验证（True 时匹配更严格但更少）

# --- ORB 匹配器（汉明距离）---
bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# --- SIFT 匹配器（欧氏距离）---
bf_sift = cv.BFMatcher(cv.NORM_L2, crossCheck=False)


# ---------- 5. Lowe's Ratio Test（比率测试筛选） ----------
# 原理：
#   对每个描述子，找到最近的两个匹配（k=2）
#   如果最近匹配的距离远小于次近匹配（< 0.75 倍），
#   说明这个匹配是"独特"的，质量高
#   如果两个距离差不多，说明这个特征点在另一张图中有多个相似区域，不可靠
#
# 为什么用 0.75？
#   Lowe 论文中实验表明 0.7-0.8 是最佳阈值
#   越小越严格（保留更少但更准确的匹配）
#   越大越宽松（保留更多但可能有误匹配）

# knnMatch(des1, des2, k=2)
#   对 des1 中的每个描述子，在 des2 中找 k 个最近邻
#   返回值: 每个元素是一组 k 个 DMatch 对象
#     DMatch.distance: 匹配距离（越小越好）
#     DMatch.queryIdx: 查询描述子在 des1 中的索引
#     DMatch.trainIdx: 匹配描述子在 des2 中的索引

# --- ORB 匹配筛选 ---
matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
good_orb = []  # 存放通过筛选的优质匹配
for m, n in matches_orb:
    # m: 最近匹配, n: 次近匹配
    # 如果 m 的距离 < n 的距离 × 0.75，认为 m 是可靠匹配
    if m.distance < 0.75 * n.distance:
        good_orb.append(m)
print(f"ORB  优质匹配点数: {len(good_orb)}")

# --- SIFT 匹配筛选 ---
matches_sift = bf_sift.knnMatch(des1, des2, k=2)
good_sift = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        good_sift.append(m)
print(f"SIFT 优质匹配点数: {len(good_sift)}")


# ---------- 6. 估计单应矩阵（Homography） ----------
# 单应矩阵 H 是一个 3×3 矩阵，描述了两张图之间的透视变换关系：
#   [x']   [h11 h12 h13] [x]
#   [y'] = [h21 h22 h23] [y]
#   [1 ]   [h31 h32  1 ] [1]
#
# 用途：将左图的像素坐标映射到右图的坐标系
# 要求：至少 4 对匹配点（因为 H 有 8 个自由度）

if len(good_sift) >= 4:
    # --- 6a. 提取匹配点坐标 ---
    # m.queryIdx: 左图关键点 kp1 中的索引
    # m.trainIdx: 右图关键点 kp2 中的索引
    # kp1[m.queryIdx].pt: 左图关键点的 (x, y) 坐标
    #
    # reshape(-1, 1, 2) 的含义：
    #   -1: 自动计算该维度大小（= 匹配点数量）
    #   1:  OpenCV 要求的中间维度
    #   2:  每个点有 x, y 两个坐标
    # 最终形状: (N, 1, 2)，N 是匹配点数量
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

    # --- 6b. RANSAC 估计单应矩阵 ---
    # cv.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)
    #   srcPoints: 源点（左图匹配点坐标）
    #   dstPoints: 目标点（右图匹配点坐标）
    #   method:    RANSAC（随机采样一致性），自动剔除离群点（错误匹配）
    #   ransacReprojThreshold: 重投影误差阈值（像素）
    #     - 如果一个点变换后的投影与真实点的距离 > 此值，认为是离群点
    #     - 5.0 是常用值，越大越宽容，越小越严格
    #   返回值:
    #     H:    3×3 单应矩阵
    #     mask: 内点掩码（形状 (N, 1)），1=内点（正确匹配），0=离群点（错误匹配）
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=5.0)

    # 将 mask 从 (N, 1) 展平为一维列表，用于后续可视化
    matches_mask = mask.ravel().tolist()

    # 统计内点数量（正确匹配数）
    inlier_count = sum(matches_mask)
    print(f"SIFT 内点数: {inlier_count} / {len(good_sift)}")


    # ---------- 7. 透视变换并拼接 ----------
    # 目标：将左图 img_1 透视变换到右图 img_2 的坐标系，然后拼接

    # 获取两张图的尺寸
    h1, w1 = img_1.shape[:2]  # 左图的 (高度, 宽度)
    h2, w2 = img_2.shape[:2]  # 右图的 (高度, 宽度)

    # --- 7a. 计算变换后左图的四个角点 ---
    # 左图的四个角点：左上(0,0), 左下(0,h1), 右下(w1,h1), 右上(w1,0)
    corners_img1 = np.float32([
        [0, 0],      # 左上角
        [0, h1],     # 左下角
        [w1, h1],    # 右下角
        [w1, 0]      # 右上角
    ]).reshape(-1, 1, 2)

    # cv.perspectiveTransform(src, H)
    #   将 src 中的点用单应矩阵 H 做透视变换
    #   返回变换后的坐标（在右图坐标系中的位置）
    warped_corners = cv.perspectiveTransform(corners_img1, H)

    # --- 7b. 计算输出画布大小 ---
    # 右图的四个角点（在右图自己的坐标系中，不需要变换）
    corners_img2 = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    # 将变换后的左图角点和右图角点合并
    all_corners = np.concatenate((warped_corners, corners_img2), axis=0)

    # 找到所有角点的最小/最大坐标，确定画布范围
    #   min(axis=0): 沿第 0 轴（行）取最小值，得到 [x_min, y_min]
    #   ravel(): 将 (1, 2) 展平为 (2,)
    #   ±0.5: 留一点边距
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # --- 7c. 构造平移矩阵 ---
    # 因为变换后的坐标可能是负数（左图可能跑到画布左边外面）
    # 需要整体平移，使所有坐标为正
    # tx = -xmin: 水平平移量（把最左的点移到 x=0）
    # ty = -ymin: 垂直平移量（把最上的点移到 y=0）
    translation = [-xmin, -ymin]

    # 平移矩阵 T（3×3）：
    #   [1  0  tx]    tx: 水平平移量
    #   [0  1  ty]    ty: 垂直平移量
    #   [0  0   1]
    Tx = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    # 最终变换矩阵 = 平移 × 单应矩阵
    # 先做透视变换（H），再做平移（Tx）
    # 矩阵乘法顺序：Tx @ H（先 H 后 Tx）
    final_H = Tx.dot(H)

    # --- 7d. 执行透视变换 ---
    # cv.warpPerspective(src, M, dsize)
    #   src: 输入图像
    #   M:   3×3 变换矩阵
    #   dsize: 输出图像尺寸 (宽度, 高度) —— 注意是 (宽, 高)，不是 (高, 宽)
    #   返回值: 变换后的图像
    result_size = (xmax - xmin, ymax - ymin)  # (宽, 高)
    stitched = cv.warpPerspective(img_1, final_H, result_size)

    # --- 7e. 将右图放置到画布上 ---
    # stitched 是左图变换后的结果，现在把右图"贴"到对应位置
    # 切片赋值：
    #   stitched[y:y+h2, x:x+w2] 表示画布上从 (x, y) 开始的 h2×w2 区域
    #   = img_2 把右图的像素填入这个区域
    stitched[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = img_2


    # ---------- 8. 可视化匹配结果 ----------
    # cv.drawMatches(img1, kp1, img2, kp2, matches, outImg, **params)
    #   img1, kp1:   第一张图及其关键点
    #   img2, kp2:   第二张图及其关键点
    #   matches:     匹配列表（DMatch 对象）
    #   outImg:      输出图像（None 则自动创建）
    #   params:
    #     matchColor:       匹配线的颜色 (B, G, R)
    #     singlePointColor: 未匹配关键点的颜色（None = 不画）
    #     matchesMask:      掩码列表，只绘制 mask[i]=1 的匹配
    #     flags:            绘制标志（2 = 不画单个关键点，只画匹配线）
    draw_params = dict(
        matchColor=(0, 255, 0),     # 绿色
        singlePointColor=None,
        matchesMask=matches_mask,    # 只画内点（正确匹配）
        flags=2
    )
    match_img = cv.drawMatches(img_1, kp1, img_2, kp2, good_sift, None, **draw_params)

    # 显示结果
    img_show('SIFT 匹配（绿线=内点）', match_img)
    img_show('拼接结果', stitched)

else:
    # 匹配点不足 4 个，无法估计单应矩阵
    print(f"SIFT 匹配点太少：{len(good_sift)}（需要至少 4 个），无法计算单应矩阵。")
    print("尝试显示 ORB 匹配结果作为参考：")

    # 退路：显示 ORB 的匹配结果
    if len(good_orb) > 0:
        match_img_orb = cv.drawMatches(
            img_1, kp1_orb, img_2, kp2_orb, good_orb, None,
            flags=2  # 只画匹配线，不画单个关键点
        )
        img_show('ORB 匹配', match_img_orb)
    else:
        print("ORB 也没有优质匹配，请检查两张图是否有足够重叠区域。")
