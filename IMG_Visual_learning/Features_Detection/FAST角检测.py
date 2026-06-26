#------------------FAST角检测------------------#
#FAST（Features from Accelerated Segment Test）
# 是一种快速的角点检测算法，主要用于计算机视觉中的特征提取。
# 它通过比较像素值来检测图像中的角点，具有较高的计算效率。

'''#流程与SIFT类似：
1. 读取图像并转换为灰度图。
2. 使用FAST算法检测角点。
3. 对检测到的角点进行非极大值抑制以获得更准确的结果。
4. 可选：使用亚像素级角点检测（如cornerSubPix）来进一步精确角点位置。
'''

import cv2 as cv
from matplotlib import pyplot as plt

# 读取图像
img = cv.imread('Features_Detection\\BB1.jpg')
assert img is not None, f"图像加载失败，请检查路径 {img} 是否正确。"

# 将图像转换为灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 创建FAST检测器对象
fast = cv.FastFeatureDetector_create(
    threshold=40,          # 阈值，较高的值会减少检测到的角点数量
    nonmaxSuppression=True, # 是否启用非极大值抑制
    type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16  # 邻域类型，常用的有9-16、7-12等
)

# 检测角点
keypoints = fast.detect(gray, None)

#检测关键点：
kp = fast.detect(gray, None)
kp_count_with_nonmax = len(kp)
# 绘制关键点:
img_kp = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

#打印所有的默认参数：
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imwrite('Features_Detection\\FAST_keypoints.jpg', img_kp)

## Disable nonmaxSuppression移除非极大值抑制处理：
fast.setNonmaxSuppression(0)
kp = fast.detect(gray, None)
kp_count_without_nonmax = len(kp)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img_kp_nonmax = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imwrite('Features_Detection\\FAST_keypoints_nonmax.jpg', img_kp_nonmax)

# 并排显示非极大值抑制前后结果
img_kp_rgb = cv.cvtColor(img_kp, cv.COLOR_BGR2RGB)
img_kp_nonmax_rgb = cv.cvtColor(img_kp_nonmax, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f'With nonmaxSuppression ({kp_count_with_nonmax} keypoints)')
plt.imshow(img_kp_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Without nonmaxSuppression ({kp_count_without_nonmax} keypoints)')
plt.imshow(img_kp_nonmax_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()

# 结论：
# 1. 启用非极大值抑制后，检测到的角点数量明显减少，但质量更高，通常更适合后续的特征匹配等任务。
# 2. 禁用非极大值抑制会检测到更多的角点。
#       但其中很多可能是重复或质量较差的点，可能会对后续处理产生负面影响。 
#  
# 因此，在实际应用中：
#   通常建议启用非极大值抑制以获得更可靠的角点检测结果。    



