'''只有匹配功能，没有输出，读写，显示画图功能'''
'''只取第二层效果并不是很好'''
import csvfile as csvfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import getcwd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
import re
import cut
import time



def judge_2(min1, cutnum):
    '''
    功能：第二步判断是否匹配，判断模拟图是否在原图中匹配到原来对应的附近位置，容许偏差阈值+-3
    :param min1: 模拟图左上坐标
    :param noise_p: 模拟图路径
    :return:匹配成功=True；匹配失败=False
    '''
    number = cutnum
    if (number % 12) == 0:
        xmin = 176
        ymin = int(number / 12 - 1) * 16  # 0-198
    else:
        xmin = (number % 12 - 1) * 16  # 0-198
        ymin = int(number / 12) * 16  # 0-198
    # print('xmin = ' + str(xmin))
    # print('ymin = ' + str(ymin))
    if within_range(min1[0], xmin, min1[1], ymin):
        return True
    else:
        return False


def judge_1(trainIndex):
    '''
    功能：第一步判断是否匹配：对于一个特征点会不会对应另一张图上多个特征点，如果出现这样的情况那么说明匹配失败
    :param trainIndex: 模拟图中的健壮sift特征点的索引
    :return: 如果相同则返回False, 如果没有不同则返回True
    '''
    # print(trainIndex)
    if len(trainIndex) == 1:
        return True
    else:
        for i in range(0, len(trainIndex) - 1):
            if trainIndex[i] == trainIndex[i + 1]:
                return False
            else:
                return True


def within_range(min1x, xmin, min1y, ymin):
    '''
    功能：判断模拟图的左上坐标是否在原图中响应位置附近，容许正负偏差阈值+-3
    :param min1x: 模拟图左上坐标中x值
    :param xmin: 原图左上坐标中x值
    :param min1y: 模拟图左上坐标中y值
    :param ymin: 原图左上坐标中y值
    :return: True, False
    '''
    if abs(min1x - xmin) <= 5 and abs(min1y - ymin) <= 5:
        return True
    else:
        return False


def get_rect(res, kp, goodnum, index):
    '''
    功能：根据健壮的sift特征点获取坐标，并且取得中心点
    KeyPoint_convert：从keypoint结构的值中根据queryIdx（原图中sift点的索引）提取出坐标
    np.ravel：多维压缩成一维取得扁平值作为中心点
    :param res:原始sift的匹配对（成对）
    :param kp:sift关键点
    :param goodnum:健壮sift的个数
    :param index:0=原图；1=模拟图
    :return: minpoint, maxpoint分别代表中心点的x,y坐标
    '''
    point_img = []
    for i in res[:goodnum]:
        if index == 0:
            center = cv2.KeyPoint_convert(kp, keypointIndexes=[i.queryIdx])
            # print(center)
        elif index == 1:
            center = cv2.KeyPoint_convert(kp, keypointIndexes=[i.trainIdx])
        center = [int(np.ravel(center)[0]), int(np.ravel(center)[1])]
        point_img.append(center)
    minres = np.argmin(point_img, axis=0)
    maxres = np.argmax(point_img, axis=0)
    minpoint = [point_img[minres[0]][0], point_img[minres[1]][1]]
    maxpoint = [point_img[maxres[0]][0], point_img[maxres[1]][1]]
    return minpoint, maxpoint


def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = -128 | octave
    # scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    # return octave
    return layer


def BFmatch(cutnum, distance, imgnum):
    '''
    功能：判断模拟图是否和原图匹配，经过3次判断:（1）模拟图不能提取到有效sift特征点（2）出现一对多（3）特征点不稳定（欧氏距离>阈值）
    :param ori_p: 原图路径
    :param cutnum: 第k张模拟图
    :param distance: 欧式距离，判定sift匹配对是否健壮：<140为十分健壮，<200为相对健壮
    :return: 匹配成功=True；匹配失败=False
    '''
    ori_img = cv2.imread(ori_path + str(imgnum)+ori_form)
    noise_img = cv2.imread(cut_path + str(imgnum) + r'/cut' + str(imgnum) + '_' + str(cutnum) + cut_form)
    hessian = 100  # 取100个初始sift特征点
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.03,  # 去除对比度低的点  DOG空间极值检测
        edgeThreshold=10,  # 去除不稳定的边缘响应点
        sigma=1.6
    )
    keypoints_ori, descriptor_ori = sift.detectAndCompute(ori_img, None)
    keypoints_noise, descriptor_noise = sift.detectAndCompute(noise_img, None)
    kp_second_ori = []
    dt_second_o = []
    kp_second_noise = []
    dt_second_n = []
    for i in range(0, len(keypoints_ori)):
        if unpackOctave(keypoints_ori[i]) == 2:
            kp_second_ori.append(keypoints_ori[i])
            dt_second_o.append(descriptor_ori[i])
    for j in range(0, len(keypoints_noise)):
        if unpackOctave(keypoints_noise[j]) == 2:
            kp_second_noise.append(keypoints_noise[j])
            dt_second_n.append(descriptor_noise[j])
    dt_second_ori = np.array(dt_second_o)
    dt_second_noise = np.array(dt_second_n)

    if len(tuple(kp_second_noise)) == 0:
        print('匹配失败')
        return False
    else:
        bf = cv2.BFMatcher()
        res = bf.match(dt_second_ori, dt_second_noise)
        res = sorted(res, key=lambda x: x.distance)
        good = []  # 这里主要是匹配的更加精确，可以修改ratio值得到不同的效果，即下面的0.75
        trainindex = []
        queryindex = []
        for m in res:
            if m.distance < distance:
                good.append([m])
                trainindex.append(m.trainIdx)
                queryindex.append(m.queryIdx)
        good_num = len(trainindex)
        print('第' + str(cutnum) + '张模拟图片匹配')
        print('足够健壮的sift匹配对个数 = ' + str(good_num))
        if good_num == 0:
            print('匹配失败')
            return False
        else:
            min1, max2 = get_rect(res, tuple(kp_second_ori), good_num, 0)
            min3, max4 = get_rect(res, tuple(kp_second_noise), good_num, 1)
            min1[0] = min1[0] - min3[0]
            min1[1] = min1[1] - min3[1] + 1
            max2[0] = min1[0] + 64
            max2[1] = min1[1] + 64
            # print('通过sift匹配出来的模糊子图在原图的位置：左上坐标（右下坐标就是左上的基础上xy各加64）')
            # print(min1)

            if judge_1(trainindex):
                if judge_2(min1, cutnum):
                    print('匹配成功')
                    return True
                else:
                    print('匹配失败')
                    return False
            else:
                print('匹配失败')
                return False



def count(cut_num, distance, imgnum):
    '''
    功能：144张模拟图和原图匹配并且计数，得出匹配概率。
    :param ori_p: 原图路径
    :param cut_num: 模拟子图数量（144）
    :param distance: 欧式距离，越小SIFT匹配对越健壮，一般去取140-250
    :return: 返回匹配概率ratio
    '''
    correctnum = 0
    for i in range(1, cut_num + 1):
        if BFmatch(i, distance, imgnum):
            correctnum += 1
    print('此原图共匹配成功' + str(correctnum) + '张')
    ratio = correctnum / 144
    print('此原图匹配概率 = ' + str(ratio))
    return imgnum,ratio


def count_all(cut_num, distance, count_num):
    ratio_list = [[0 for m in range(2)] for n in range(0, count_num)]
    for i in range(1, count_num):  # 1697
        ratio_list[i-1][0],ratio_list[i-1][1] = count(cut_num, distance, i)
    print(ratio_list)
    ratio_list = sorted(ratio_list[:count_num], key=lambda x:x[1], reverse=True)
    with open(r'./result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['imgnum', 'ratio'])
        writer.writerows(ratio_list)
        print('result updated')


if __name__ == '__main__':
    ori_path = r'./originopt/opt'
    ori_form = '.png'
    cut_path = r'./cut/cut'
    cut_form = '.png'

    start_time = time.perf_counter()
    count_all(144, 180, 2)
    end_time = time.perf_counter()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print('运行时间 = ' + str(run_time))
