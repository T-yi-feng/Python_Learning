#---------模板Template匹配---------#
#模板匹配就是：
#在大图中寻找和小图相似的区域。


import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

img1 = cv2.imread('img1.png',cv2.IMREAD_GRAYSCALE)  #读取大图
assert img1 is not None, "请检查图片路径是否正确"


template = cv2.imread('part_of_img1.png',cv2.IMREAD_GRAYSCALE)  #读取小图
assert template is not None, "请检查模板子图片路径是否正确"

#获取小图的宽和高：
# template.shape 返回 (height, width)。
# 使用 [::-1] 对元组进行反向切片，步长为 -1 表示倒序，
# 从而得到 (width, height)，再赋值给 w,h
w,h = template.shape[::-1]

##一共有六种常见的模板匹配方法：
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED',
            'TM_CCORR','TM_CCORR_NORMED',
              'TM_SQDIFF', 'TM_SQDIFF_NORMED']

#分别为：中文名：
methods_chinese = ['相关系数匹配', '归一化相关系数匹配',
                   '相关匹配','归一化相关匹配',
                   '平方差匹配', '归一化平方差匹配']

for method, method_chinese in zip(methods, methods_chinese):
    img1_copy = img1.copy()  #复制大图，避免在原图上进行绘制，影响后续的匹配结果
    #✨将字符串转换为cv2库中的方法对象：
    method_obj = getattr(cv2, method)

    #apply template Matching
    res = cv2.matchTemplate(img1_copy,template,method_obj)
    #res是一个二维数组，表示每个位置的匹配结果
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #获取匹配结果中的最小值、最大值及其位置

    #根据不同的匹配方法，选择合适的匹配位置：
        #如果是TM_SQDIEFF或TM_SQDIFF_NORMED方法，匹配结果越小越好，选择最小值：
    if method_obj in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    #计算匹配区域的右下角坐标：
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img1_copy,top_left, bottom_right, 255, 2)

    plt.subplot(121)
    plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img1_copy,cmap = 'gray')
    plt.title('观测特征点')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(f'{method} ({method_chinese})')
    plt.show()

#实际上的结果会被很大程度上的无关白色值所影响。
#导致模板匹配会匹配到白天上的云，而不是地面上的特征点。














