#-----------哈里斯角检测CornerHarris------------#
#✨函数：
# OpenCV有函数cv.cornerHarris()用于此目的。其参数如下：
#   img - 输入图像。应该是灰度和float32类型。
#   blockSize - 它是角检测中考虑的邻域大小
#   ksize - 所用索贝尔导数的孔径参数。
#   k - 方程中的哈里斯探测器自由参数。
import cv2 as cv
import numpy as np
import os
from PIL import Image

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

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取图像 - 使用绝对路径和自定义函数
img_path = os.path.join(script_dir, 'OIP.jpg')
img = imread_chinese_path(img_path)
if img is None:
    raise FileNotFoundError(f'无法读取图片: {img_path}')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 问题2：cornerHarris 要求输入图像是 float32，这里转换是正确的，保留即可。
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

# 问题3：dilate 的 kernel 不能传 None（静态类型检查会报重载不匹配），需要显式传入结构元素。
kernel = np.ones((3, 3), np.uint8)
dst = cv.dilate(dst, kernel)

## Threshold for an optimal value, it may vary depending on the image.
# 问题4：下面的阈值逻辑本身没有语法问题，保留原写法，仅修复前面的调用错误。
img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


#-----------亚像素级角点检测CornerSubPix------------#
filename = 'Features_Detection\\OIP.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

## find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

## find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

## define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

## Now draw them
res = np.hstack((centroids,corners))
res = res.astype(int)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv.imwrite('Features_Detection/subpixel5.png',img)


















