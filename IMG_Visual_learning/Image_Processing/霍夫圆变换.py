#-------------霍夫圆变换Hough Circle Transform----------------#
#✨函数：
#   cv.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
#   image - 8位，单通道，灰度输入图像。
#   method - 检测方法，目前唯一实现的是 cv.HOUGH_GRADIENT。
#   dp - 累加器分辨率与图像分辨率的反比。dp=1表示累加器与图像具有相同的分辨率。
#   minDist - 圆心之间的最小距离。较小的值可能会导致多个邻近圆被错误地检测为一个圆。
#   circles - 输出参数，检测到的圆的矢量，每个圆由三个值组成：圆心的x坐标、圆心的y坐标和圆的半径。
#   param1 - 方法特定的参数。对于 cv.HOUGH_GRADIENT，它
#             是传递给 Canny 边缘检测器的高阈值。低阈值是高阈值的一半。
#   param2 - 方法特定的参数。对于 cv.HOUGH_GRADIENT，它是累加器阈值，只有累加器值大于该阈值的圆才被检测到。
#   minRadius - 圆半径的最小值。比这个更小的圆被拒绝。
#   maxRadius - 圆半径的最大值。比这个更大的圆被拒绝。
# 返回值：如果检测到圆，返回一个包含圆参数的 numpy.ndarray
# 
# 每个圆由三个值组成：
#               圆心的x坐标
#               圆心的y坐标
#               圆的半径；
#               否则返回 None。


# https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html






