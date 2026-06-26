#------------霍夫线HoughLines变换----------------#

#知识点参考：https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#这个讲的很好

import cv2 as cv 
import numpy as np

img2 = cv.imread('img2.jpg')  #读取图片
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  #转为灰度图
edges = cv.Canny(gray, 50, 150, apertureSize=3)  #边缘检测


# 参数说明（cv.HoughLines）：
# 第1个参数 image: 输入的二值边缘图（一般用 Canny 的输出），数据类型为 8 位单通道。
# 第2个参数 rho: 距离分辨率，以像素为单位。通常取 1 表示精度为 1 像素。
# 第3个参数 theta: 角度分辨率，以弧度为单位。常用 np.pi/180（即 1 度）。
# 第4个参数 threshold: 阈值，累加平面中>阈值的点被认为是一条直线（越大检测到的直线越少、越显著）。
# 可选参数：srn, stn（用于多尺度霍夫变换，默认 0）；min_theta, max_theta（角度范围，单位为弧度，默认 0 到 pi）。
# 返回值：如果检测到直线，返回形如 (N,1,2) 的 ndarray，每个元素为 [[rho, theta]]；否则返回 None。
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
# 霍夫变换检测直线
# 返回值说明：
# - 如果检测到直线，lines 是一个 numpy.ndarray，形状通常为 (N, 1, 2)
#   每个元素为 [[rho, theta]]，其中 rho 和 theta 是浮点数，分别表示
#   极坐标下的直线距离和角度（theta 以弧度为单位）。例如：
#   array([[[ 100.        ,    1.57079633]],
#          [[ 200.        ,    0.78539816]],
#          ...], dtype=float32)
# - 如果未检测到直线，lines 为 None。
# 常见的遍历方式：
# if lines is not None:
#     for rho, theta in lines[:, 0]:
#         # 使用 rho, theta 绘制或处理直线
#         pass

# 示例安全遍历：
# if lines is not None:
# 	for rho, theta in lines[:, 0]:
# 		print('rho={}, theta={}'.format(rho, theta))

for line in lines:
    rho,theta = line[0]  #获取每条线的rho和theta
    cos = np.cos(theta)  #计算theta的余弦值
    sin = np.sin(theta)  #计算theta的正弦值
    x0 = rho * cos  #计算线段的起点坐标x0
    y0 = rho * sin  #计算线段的起点坐标y0
    x1 = int(x0 + 1000 * (-sin))  #计算线段的终点坐标
    y1 = int(y0 + 1000 * (cos))  #计算线段的终点坐标
    x2 = int(x0 - 1000 * (-sin))  #计算线段的终点坐标
    y2 = int(y0 - 1000 * (cos))  #计算线段的终点坐标

    cv.line(img2, (x1, y1), (x2, y2), (255, 255, 12), 1)  #在原图上绘制红色线段
    
cv.imwrite('houghlines_img2.png', img2)  #保存结果图像
cv.imshow('houghlines_img2', img2)  #显示结果图像
cv.setWindowTitle('houghlines_img2', 'Hough Lines Detected')  #设置窗口标题
cv.waitKey(0)

#这种方法稍微复杂点的现实图片，就没啥用了.
#而且有个缺点就是：一条直线会有很多点满足条件，所以会有很多重复的线段被检测出来。


#接下来是概率霍夫变换：
#------------概率霍夫线Probablistic Hough Transform----------------#
#✨函数：
#   cv.HoughLinesP(image, rho, theta, threshold, minLineLength=0, maxLineGap=0)

# 它提出了两个新的论点：

#        minLineLength - 最小线长度。比这个更短的线段被拒绝。

#        maxLineGap - 线段之间允许的最大间隙，以便将它们视为单线。

# 最好的是，它直接返回了两条线的端点。
# 在之前的情况中，你只有line的参数，你必须找到所有点.

img = cv.imread('img2.jpg')  #读取图片
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #转为灰度
edges = cv.Canny(gray, 50, 150, apertureSize=3)  #边缘检测
cv.imshow('edges', edges)  #显示边缘图像
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=10)  #概率霍夫变换检测线段

#由于概率霍夫变换直接提供了两条线的端点，则无需进行额外的计算：
for line in lines:
    x1,x2,y1,y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (255, 255, 12), 1)  #在原图上绘制红色线段

cv.imwrite('Probabilistic_Hough_Lines_img2.png', img)  #保存结果图像
cv.imshow('Probabilistic_Hough_Lines_img2', img)  #显示结果图像
cv.setWindowTitle('Probabilistic_Hough_Lines_img2', 'Probabilistic Hough Lines Detected')  #设置窗口标题
cv.waitKey(0)

#但是结果太不稳定了，甚至有时候效果不如普通霍夫变换，可能是参数设置不当导致的。
#😂

