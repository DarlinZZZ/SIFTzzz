'''只有匹配功能，没有输出，读写，显示画图功能'''
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


def judge_2(center_ori, center_noise, xmin, ymin):
    N = len(center_ori)  # 一共多少匹配对
    c = []
    for i in range(N):
        c.append([[elem[0] - xmin, elem[1] - ymin] for elem in center_ori[i]])
    center = np.array(c)

    # print(center)

    # 计算arr1和arr2中所有元素的两两距离
    distances = np.sqrt(np.sum((center[:, np.newaxis, :] - center_noise) ** 2, axis=-1))
    # print(distances)

    # 找出最小距离对应的索引
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    # 最小距离
    min_distance = distances[min_index]
    # print(min_index, min_distance)

    N1 = 0
    if len(distances) != 0:
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if distances[i][j] < np.sqrt(18):
                    N1 += 1

    if min_distance > np.sqrt(18):
        # print("匹配失败")  # 表明没有一个特征点对是符合要求的
        return False
    elif min_distance < np.sqrt(18) and N1 / N > 0.5:
        return True
        # print("匹配成功")
        print('健壮匹配对有 ' + str(N1) + ' 对')
    else:
        return False
        # print("匹配失败")
        print('健壮匹配对有 ' + str(N1) + ' 对')


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


def get_rect(res, kp, goodnum, index):
    center = []
    for i in res[:goodnum]:
        if index == 0:
            center.append(cv2.KeyPoint_convert(kp, keypointIndexes=[i.queryIdx]))
        elif index == 1:
            center.append(cv2.KeyPoint_convert(kp, keypointIndexes=[i.trainIdx]))
    return center


def get_cutxy(number):
    if (number % 12) == 0:
        xmin = 176
        ymin = int(number / 12 - 1) * 16  # 0-198
    else:
        xmin = (number % 12 - 1) * 16  # 0-198
        ymin = int(number / 12) * 16  # 0-198
    return xmin, ymin



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
    if len(keypoints_noise) == 0:
        print('匹配失败')
        return False
    else:
        bf = cv2.BFMatcher()
        res = bf.match(descriptor_ori, descriptor_noise)
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


        number = int(cutnum)
        xmin, ymin = get_cutxy(number)

        print('足够健壮的sift匹配对个数 = ' + str(good_num))
        if good_num == 0:
            print('匹配失败')
            return False
        else:
            center_ori = np.array(get_rect(res, keypoints_ori, good_num, 0))

            center_noise = np.array(get_rect(res, keypoints_noise, good_num, 1))


            if judge_1(trainindex):
                if judge_2(center_ori, center_noise, xmin, ymin):
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
    with open(r'./result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['imgnum', 'ratio'])
        writer.writerows(ratio_list)
        print('result updated')
    ratio_list_sorted = sorted(ratio_list[:count_num], key=lambda x:x[1], reverse=True)
    with open(r'./result_sorted.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['imgnum', 'ratio'])
        writer.writerows(ratio_list_sorted)
        print('result_sorted updated')


if __name__ == '__main__':
    ori_path = r'./originopt/opt'
    ori_form = '.png'
    cut_path = r'./cut/cut'
    cut_form = '.png'

    start_time = time.perf_counter()
    count_all(144, 180, 1601)
    end_time = time.perf_counter()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print('运行时间 = ' + str(run_time))
