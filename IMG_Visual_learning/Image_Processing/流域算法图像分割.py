#-----------使用流域算法Watershed进行图像分割-----------------#
# 任何灰度图像都可以被视为一个地形表面。
# 其中高强度表示峰值和丘陵，而低强度则表示山谷。
# 你开始填满每个孤立的谷地（局部极小值）用不同颜色的水（标签）。
# 随着水位上升，取决于山峰（坡度） 附近，不同山谷的水，显然颜色不同，将开始汇合。
# 你继续注水并建造屏障，直到所有山峰都被淹没。
# 最终，屏障的线条将形成分割图像的边界。


import cv2
import numpy as np

img = cv2.imread('water_coins.jpg')  #读取图片
assert img is not None, "Image not found!"  #确保图片被正确读取
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰度化

#将图片进行二值化：
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  

#去除噪声：
kernel = np.ones((3,3),np.uint8)  #定义一个3x3的卷积核
#在这里我们使用开运算（先腐蚀后膨胀）来去除小的噪声点：
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)

#两个结果一起展示
show = np.hstack((thresh, opening))
cv2.imshow('Binary Image | Opening', show)
cv2.waitKey(0)

#接下来：
# 扩大硬币，让背景的黑色区域变得更小，使得识别出来的背景一定是背景:
sure_bg = cv2.dilate(opening,kernel,iterations=3)
#对距离变换结果进行二值化，得到确定的背景区域:
# 缩小硬币，让背景的黑色区域变得更大，使得识别出来的硬币一定是硬币:
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)  
#对距离变换结果进行二值化，得到确定的前景区域:
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv2.imshow('Distance Transform', sure_fg.astype(np.uint8))
cv2.imshow('Sure Background', sure_bg)
# 将前景区域转换为uint8类型，并找到不好区分的区域:
sure_fg = np.uint8(sure_fg)
sure_bg = np.uint8(sure_bg)
unknown = cv2.subtract(sure_bg, sure_fg)

cv2.imshow('Unknown Region', unknown)
cv2.waitKey(0)

#至此，我们已经得到了确定的前景区域、确定的背景区域和未知区域。
# 接下来，我们需要为每个连通组件分配一个唯一的标签。
# 以便在后续的分水岭算法中使用：
#使用：connectedComponents函数为每个连通组件分配一个唯一的标签：

#mMarker labelling:
# connectedComponents() 返回两个值: 
#       ret 是连通组件的数量 (包含背景),
#       markers 是每个像素的标签图 (uint32/ int), 背景通常为0, 前景从1开始。
ret, markers = cv2.connectedComponents(sure_fg)

# 为了确保背景的标签为0，我们将所有标签加1:
# (因为我们设定需要不确定的面积都保持为0，但是markers中已经有0作为背景了
# 所以我们需要将所有标签加1，使得背景仍然是0，前景从1开始)
markers = markers + 1   

markers[unknown==255] = 0  # 将未知区域的标签设置为0

# 为了更直观地展示每个marker，给每个标签分配一种不同的颜色并显示：
# 注意：markers中0表示未知/边界，我们用黑色表示
np.random.seed(42)
labels = np.unique(markers)
colored_markers = np.zeros_like(img)
for lbl in labels:
	if lbl == 0:
		colored_markers[markers == lbl] = (0, 0, 0)
	else:
		color = tuple(np.random.randint(0, 256, size=3).tolist())
		colored_markers[markers == lbl] = color

cv2.imshow('Markers before Watershed (colored)', colored_markers)
cv2.waitKey(0)

#最后，我们使用分水岭算法进行图像分割：
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 255]  # 将分水岭边界标记为红色
cv2.imshow('Watershed Result', img)
cv2.waitKey(0)




