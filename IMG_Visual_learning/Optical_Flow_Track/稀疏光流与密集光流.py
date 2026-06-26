#教程：https://learnopencv.com/optical-flow-in-opencv/

#Lucas-Kanade光流法是一种基于局部图像块的光流估计方法。
#计算稀疏光流，即只计算图像中某些特定点的光流：
#   采用的是Shi-Tomasi角点检测算法来选择特征点
#   然后使用Lucas-Kanade方法来跟踪这些特征点在连续帧中的位置变化。


import cv2 as cv
import numpy as np

def LucasMethod(vedio_path):
    cap = cv.VideoCapture(vedio_path)

    #定义Shi-Tomasi角点检测参数：
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # maxCorners: 最多检测的角点数量，限制特征点个数
    # qualityLevel: 角点质量阈值，数值越大，筛选越严格
    # minDistance: 角点之间的最小距离，避免特征点过于密集
    # blockSize: 计算角点时使用的邻域窗口大小

    #定义Lucas-Kanade光流法参数：
    lk_params = dict(
        # 搜索窗口大小，光流在每个金字塔层对每个特征点搜索位移的区域
        # （越大能捕捉更大位移，但计算更慢）
        winSize=(15, 15),
        # 金字塔的最大层数（0表示不使用图像金字塔，>0时使用金字塔以处理大位移）
        maxLevel=2,
        # 迭代终止条件：这里使用eps和count的组合，(type, maxCount, epsilon)
        # type: 终止条件的类型；maxCount: 最大迭代次数；epsilon: 精度阈值（当更新小于该值时停止）
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    #随机颜色：
    color = np.random.randint(0, 255, (100, 3))

    #找到第一帧图像，并检测特征点：
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频或视频为空")
        return
    # 将第一帧转换为灰度图像以供后续处理
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    #使用cv.goodFeaturesToTrack()函数检测特征点：
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    #创建一个掩膜图像用于绘制轨迹：
    mask = np.zeros_like(old_frame)

    #循环读取视频帧，计算光流：
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #计算稀疏光流：
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        #选择跟踪成功的特征点：
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        #绘制轨迹：
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

        #将当前帧和掩膜叠加显示：
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:  # 按下 'ESC' 键退出循环
            break

        #更新前一帧和特征点位置：
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv.destroyAllWindows()


#密集光流（Dense Optical Flow）：
#密集光流方法计算图像中每个像素的运动矢量，而不是仅仅计算特定特征点的运动。
#常用的密集光流算法包括Farneback方法和RLOF方法。


def dense_optical_flow_gray(method, vedio_path, params=[]):
    """
    灰度图密集光流（Farneback / SparseToDense）
    输入：灰度帧，输出：HSV 可视化
    """
    cap = cv.VideoCapture(vedio_path)
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("无法读取视频或视频为空")
        cap.release()
        return

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    hsv_mask = np.zeros_like(prev_frame)
    hsv_mask[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 计算密集光流
        flow = method(prev_gray, frame_gray, None, *params)

        # 可视化：方向→色调，幅值→亮度
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv.normalize(mag, np.zeros_like(mag), 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)

        cv.imshow('Original', frame)
        cv.imshow('Dense Optical Flow', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        prev_gray = frame_gray

    cap.release()
    cv.destroyAllWindows()


def dense_optical_flow_color(method, vedio_path, max_width=640):
    """
    彩色图密集光流（RLOF）
    RLOF 要求输入 3 通道彩色图，且高分辨率会触发断言错误
    所以先缩小再计算
    """
    cap = cv.VideoCapture(vedio_path)
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("无法读取视频或视频为空")
        cap.release()
        return

    # 计算缩放比例
    h, w = prev_frame.shape[:2]
    scale = min(1.0, max_width / w)
    prev_small = cv.resize(prev_frame, None, fx=scale, fy=scale) if scale < 1.0 else prev_frame

    hsv_mask = np.zeros_like(prev_small)
    hsv_mask[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 缩小帧
        frame_small = cv.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame

        # RLOF 要求预分配 flow 数组
        h, w = prev_small.shape[:2]
        flow = np.zeros((h, w, 2), dtype=np.float32)
        method(prev_small, frame_small, flow)

        # 可视化
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv.normalize(mag, np.zeros_like(mag), 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)

        cv.imshow('Original', frame)
        cv.imshow('Dense Optical Flow', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        prev_small = frame_small

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    #choose from 'lucaskanade_sparse', 'farneback', 'sparse_to_dense', 'rlof')
    import argparse
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vedio_path = os.path.join(script_dir, "vedio.mp4")

    parser = argparse.ArgumentParser(description="光流法演示：稀疏光流 + 密集光流")
    parser.add_argument('--algorithm', type=str, default='lucaskanade_sparse',
                        choices=['lucaskanade_sparse', 'farneback', 'sparse_to_dense', 'rlof'],
                        help='选择光流算法: lucaskanade_sparse / farneback / sparse_to_dense / rlof')
    args = parser.parse_args()

    if args.algorithm == "lucaskanade_sparse":
        # Lucas-Kanade稀疏光流法
        LucasMethod(vedio_path)

    elif args.algorithm == "farneback":
        # Farneback 密集光流法（标准 OpenCV 自带，灰度图输入）
        # cv.calcOpticalFlowFarneback 参数说明：
        #   pyr_scale: 金字塔缩放比率，0.5 表示每层缩小一半
        #   levels: 金字塔层数，3 表示构建 3 层金字塔
        #   winsize: 平均窗口大小，越大对快速运动越鲁棒，但细节丢失
        #   iterations: 每层迭代次数，10 次足够收敛
        #   poly_n: 像素邻域大小，5-7 较常用，越大越平滑
        #   poly_sigma: 高斯标准差，用于平滑导数，1.1-1.5 较常用
        #   flags: 0 或 cv.OPTFLOW_FARNEBACK_GAUSSIAN
        farneback_params = [0.5, 3, 15, 10, 5, 1.1, 0]
        dense_optical_flow_gray(cv.calcOpticalFlowFarneback, vedio_path, params=farneback_params)

    elif args.algorithm == "sparse_to_dense":
        # SparseToDense 密集光流法（需要 opencv-contrib-python，灰度图输入）
        dense_optical_flow_gray(cv.optflow.calcOpticalFlowSparseToDense, vedio_path)

    elif args.algorithm == "rlof":
        # RLOF (Robust Local Optical Flow) 密集光流法
        # 比 Farneback 更鲁棒，对噪声和大位移更友好，但速度稍慢
        # 注意：RLOF 要求输入 3 通道彩色图
        dense_optical_flow_color(cv.optflow.calcOpticalFlowDenseRLOF, vedio_path)
