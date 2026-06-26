#-------Numpy中的傅里叶变换-------#
import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt

img = cv.imread('img1.png',cv.IMREAD_GRAYSCALE)

assert img is not None, "文件路径错误或文件不存在！"

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)#将结果移动到频谱图像的中心位置

magnitude_spectrum = 20*np.log(np.abs(fshift))

'''
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])
plt.show()
'''

#通过这个代码，我们可以看到输入图像的频谱图像。
# 频谱图像显示了图像中不同频率成分的强度。
#中心亮点扩散说明了图像中存在较强的低频成分，而周围的亮点则表示高频成分。
#可知：
#   图像的主要信息存在于图像的低频部分。
#   而高频部分则包含了图像的细节和纹理信息.

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # 频谱中心坐标（后续做低/高通滤波时会用到）
print(f"图像尺寸: {rows}x{cols}, 频谱中心: ({crow}, {ccol})")

fshift[crow-30:crow+30, ccol-30:ccol+30] = 0  
# 将频谱中心周围的低频成分置零（高通滤波）
#如果令值为1，则保留中心周围的低频成分，其他部分置零（低通滤波）。

f_ishift = np.fft.ifftshift(fshift)  
# 这里不是“重复处理”，而是把前面 fftshift 的“中心化”结果还原
# 说明：
# 1) fftshift：把低频移到中心，便于观察/在中心区域做滤波。
# 2) ifftshift：在做逆变换前，把频谱从“中心化坐标”搬回默认坐标。
#i意思是：inverse fft shift，即逆向的频谱中心化处理。


img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Numpy:Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Numpy:Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(img_back, cmap='gray')
plt.title('Numpy:Image after Low Pass Filter')
plt.xticks([])
plt.yticks([])

# 通过观察频谱图像，我们可以分析图像中的纹理、边缘等特征.

#----------OpenCV中的傅里叶变换----------#
# OpenCV为此提供了cv.dft（）和cv.idft（）这两个函数。
# 结果与之前相同，但有两个通道:
#       第一通道为结果的实数部分.
#       第二通道为结果的虚数部分.
img_cv = cv.imread('img1.png', cv.IMREAD_GRAYSCALE)
assert img_cv is not None, "文件路径错误或文件不存在！"


dft = cv.dft(img_cv.astype(np.float32), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 语法说明：cv.dft(src, flags=cv.DFT_COMPLEX_OUTPUT)
# - src: 输入图像，需为浮点型（这里用 astype(np.float32) 转换）。
# - flags: magnitude标志.
#       这里使用 cv.DFT_COMPLEX_OUTPUT 表示返回复数结果
#                               （2 通道：实部+虚部）
# 返回值 dft 为频域复数表示，常与 np.fft.fftshift/ifftshift 配合使用.

magnitude_spectrum_cv = 20*np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
# 这里的 cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) 
#               是计算复数的幅值（即频谱强度），
# 其中 dft_shift[:,:,0] 是实部，dft_shift[:,:,1] 是虚部。
# 通过对幅值取对数（20*np.log）可以更好地可视化频谱图像。

plt.subplot(234)
plt.imshow(img_cv, cmap='gray')
plt.title('OpenCV:Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(magnitude_spectrum_cv, cmap='gray')
plt.title('OpenCV:Magnitude Spectrum')
plt.xticks([])
plt.yticks([])


rows_cv, cols_cv = img_cv.shape
crow_cv, ccol_cv = rows_cv // 2, cols_cv // 2

## create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows_cv,cols_cv,2),np.uint8)
mask[crow_cv-30:crow_cv+30, ccol_cv-30:ccol_cv+30] = 1
#这里做的是低通滤波：
#   在频谱中心周围创建一个 60x60 的区域，并将其值设置为 1（保留低频成分），其他部分保持为 0（去除高频成分）。
#   注意这里的 mask 是一个三维数组，最后一个维度为 2，对应于复数的实部和虚部。

#apply mask and inverse DFT:
dft_shift = dft_shift * mask
f_ishift_cv = np.fft.ifftshift(dft_shift)
img_back_cv = cv.idft(f_ishift_cv) 
img_back_cv = cv.magnitude(img_back_cv[:,:,0], img_back_cv[:,:,1])

plt.subplot(236)
plt.imshow(img_back_cv, cmap='gray')
plt.title('OpenCV:Filtered Image')
plt.xticks([])
plt.yticks([])
plt.show()


plt.subplot(211)
plt.imshow(img_back, cmap='gray')
plt.title('Numpy:Image after Low Pass Filter')
plt.xticks([])
plt.yticks([])

plt.subplot(212)
plt.imshow(img_back_cv, cmap='gray')
plt.title('OpenCV:Filtered Image')
plt.xticks([])
plt.yticks([])
plt.show()















