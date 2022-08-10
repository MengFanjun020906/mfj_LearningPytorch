#导入基本库
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#1.1阈值过滤 1.1.1梯度阈值过滤
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

#打印出图片数据
origin =cv2.imread('06.jpg')
plt.imshow(origin)
#plt.show()
print(origin)

#对图片进行预处理
img =origin.copy()

img_x_thresh=abs_sobel_thresh(img,orient="x",thresh_min=30,thresh_max=200)#这里的x,30,200是需要填的

print(img_x_thresh)
plt.imshow(img_x_thresh)
# plt.show()

#
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag < mag_thresh[1])] = 1

    return binary_output

img_1=origin.copy()

mag_thresh_1=mag_thresh(img_1,sobel_kernel=3,mag_thresh=(50,100))#填入数据3,(50,100)


plt.imshow(mag_thresh_1)
# plt.show()

img_2=origin.copy()
mag_thresh_2=mag_thresh(img_1,sobel_kernel=27,mag_thresh=(50,100))#填入27，（50,100）
plt.imshow(mag_thresh_2)
# plt.show()


def hls_select(img, channel='s', thresh=(0, 255)):
    """使用HLS颜色空间的进行阈值过滤"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channel = hls[:, :, 0]
    elif channel == 'l':
        channel = hls[:, :, 1]
    else:
        channel = hls[:, :, 2]

    binary_output=np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


img_color_value=origin.copy()


s_thresh=hls_select(img,channel='l',thresh=(220,255))#填入l和（220,255）

plt.imshow(s_thresh)
# plt.show()


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)

    # calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def luv_select(img, thresh=(0, 255)):
        """使用LUV颜色空间的L(lightness亮度)通道进行阈值过滤"""
        luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        l_channel = luv[:, :, 0]
        binary_output = np.zeros_like(l_channel)
        binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
        return binary_output


def lab_select(img, thresh=(0, 255)):
    """
    Threshold LAB B-channel using exclusive lower and inclusive upper bound.
    """
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]

    # 2) Apply a threshold to the B channel
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output



def thresholding(img):
    # Compute individual thresholded images
    x_thresh=abs_sobel_thresh(img,orient='x',thresh_min=10,thresh_max=230)
    m_thresh=mag_thresh(img,sobel_kernel=27,mag_thresh=(70,100))
    dir_thresh = dir_threshold(img,sobel_kernel= 27, thresh=(0.7, 1.3))
    hls_thresh= hls_select(img, thresh=(180, 255))
    lab_thresh= lab_select(img,thresh=(155, 200))
    luv_thresh=luv_select(img,thresh=(225,255))

    # Compute combined threshold
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (m_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (luv_thresh == 1)] = 255
    return threshholded


img_combine =origin.copy()
comb_thresh=thresholding(img)
plt.imshow(comb_thresh)
# plt.show()

def get_M_Minv():




    scr=np.float32([[(400,720),(580,350),(780,350),(950,720)]])


    dst=np.float32([[(320,720),(320,0),(960,0),(960,720)]])

    M=cv2.getPerspectiveTransform(scr,dst)
    Minv=cv2.getPerspectiveTransform(dst,scr)
    return [M,Minv]



img_pres=origin.copy()
print(img_pres.shape)
print(img_pres.shape[1::-1])

M,Minv =get_M_Minv()

binary_warped=cv2.warpPerspective(comb_thresh,M,img_pres.shape[1::-1],flags=cv2.INTER_LINEAR)

plt.imshow(binary_warped)
plt.show()