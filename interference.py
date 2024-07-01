import cv2
import numpy as np
from skimage import util

# img_path = r'./opt1.png'
img_path = r'./originopt/opt'
# img_savepath = r'./interference/'
img_savepath = r'./noise/'

'''
ori_img: 需要滤波的图片；
(9, 9): 卷积核大小；
0.1: 高斯核函数在 X 方向的的标准偏差；(越大约模糊）
'''


def cv2_Gaussianblur(path, savepath, img_name):  # 废弃
    ori_img = cv2.imread(path)
    blur_img = cv2.GaussianBlur(ori_img, (9, 9), 0.1)
    cv2.imshow('origin img', ori_img)
    cv2.imshow('blur img', blur_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(savepath + img_name, blur_img)
    print(img_name, 'saved')


'''图片显示和保存'''


def showsave(ori_img, out_img, savepath, imgname):
    # cv2.imshow('ori_img', ori_img)
    # cv2.imshow('Gblur_img', out_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(savepath + imgname, out_img)
    # print(imgname, 'ski_saved')


'''
random_noise("gaussian")函数向浮点图像添加高斯模糊噪声。
功能：具有特定局部方差的高斯分布加性噪声在“图像”的每个点上。
步骤：1.产生一个高斯随机数2.根据输入像素计算出输出像素3.重新将像素值限制或放缩在[0 ~ 255]之间4.循环所有像素5.输出图像
Var=σ**2 随机分布的方差。用于“高斯”和“斑点”。
注:方差σ=(标准差)** 2。默认值:0.01
variance in assay = 0.36
'''


def ski_Gaussianblur(path, savepath, variance):
    ori_img = cv2.imread(path)
    Gblur_img = util.random_noise(ori_img, "gaussian", var=variance)
    Gblur_img = Gblur_img * 255
    Gblur_img = Gblur_img.astype(np.int16)
    imgname = r'Gblur_var=' + str(variance) + '.png'
    showsave(ori_img, Gblur_img, savepath, imgname)


'''
random_noise("speckle")函数向浮点图像添加乘性噪声。
功能：概述：服从image + n*image，n是具有指定均值和方差的均匀噪声
Var=σ**2 随机分布的方差。用于“高斯”和“斑点”。
注:方差σ=(标准差)** 2。默认值:0.01
variance in assay = 0.0004
'''


def ski_multinoise(path, savepath, variance):
    ori_img = cv2.imread(path)
    Mnoise_img = util.random_noise(ori_img, "speckle", var=variance)
    Mnoise_img = Mnoise_img * 255
    Mnoise_img = Mnoise_img.astype(np.int16)
    imgname = r'Mnoise_var=' + str(variance) + '.png'
    showsave(ori_img, Mnoise_img, savepath, imgname)


'''高斯模糊+乘性噪声（二值化灰度）'''


def ski_noise(path, savepath, G_variance, M_variance):
    for i in range (1,1697):  # 一共1696张原图
        ori_img_name = str(i) + '.png'
        ori_img = cv2.imread(path+ori_img_name)
        noise_img = util.random_noise(ori_img, "gaussian", var=G_variance)
        noise_img = util.random_noise(noise_img, "speckle", var=M_variance)
        noise_img = noise_img * 255
        noise_img = noise_img.astype(np.int16)
        # cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        imgname = r'ski_noise' + str(i) + '.png'
        cv2.imwrite(savepath + imgname, noise_img)
        print('ski_noise'+str(i)+' saved')
    gray(savepath + imgname, savepath, G_variance, M_variance)  # 灰度二值化处理




'''二值灰度化'''


def gray(path, savepath, G, M):
    for i in range (1,1697):  # 一共1696张原图
        ori_img_name = str(i) + '.png'
        ori_img = cv2.imread(r'./noise/ski_noise' + ori_img_name)
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        imgname = r'noise' + str(i) + '.png'
        cv2.imwrite(savepath + imgname, gray_img)
        print('noise' + str(i) + ' saved')


if __name__ == '__main__':
    # img_name = r'cv2_GB_opt1.png'
    img_name = r'opt1.png'
    # cv2_Gaussianblur(img_path, img_savepath, img_name)
    # ski_Gaussianblur(img_path, img_savepath, 0.36)
    # ski_multinoise(img_path, img_savepath, 0.0004)
    # gray(testpath, img_savepath)
    ski_noise(img_path, img_savepath, 0.005, 0.02)
