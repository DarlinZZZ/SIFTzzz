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

'''
orange - keypoints of origin img
purple - keypoints of noise img
'''

'''
BFMatcher：所有可能的匹配，寻找最佳。
FlannBasedMatcher：最近邻近似匹配，不是最佳匹配。
'''


def writekp(keypoints, kp_name):
    with open(kp_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num', 'type', 'position', 'size', 'angle', 'response', 'octave', 'class_id'])
        writer.writerow(['example = ', str(keypoints[0])])
        for i in range(0, len(keypoints)):
            writer.writerow([str(i + 1),
                             str(type(keypoints[i])),
                             str(keypoints[i].pt),
                             str(keypoints[i].size),
                             str(keypoints[i].angle),
                             str(keypoints[i].response),
                             str(unpackOctave(keypoints[i])),
                             str(keypoints[i].class_id)])
    print(kp_name + 'is updated')


def writematch(match, match_name):
    '''
    cv2.DMatch数据结构
    .distance：两个描述符的距离（越小匹配度越高）;对应特征点描述符的欧氏距离，数值越小也就说明俩个特征点越相近。
    .trainIdx：目标图像（train）中的索引值;测试图像的特征点描述符的索引（第几个特征点描述符），同时也是描述符对应特征点的索引。
    .queryIdx：训练图像（query）中的索引值（原图）;样本图像的特征点描述符索引，同时也是描述符对应特征点的索引。
    .imgIdx：目标图像的索引值（多图匹配时用到）原图=0模糊图=1
    '''
    with open(match_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num', 'type', 'distance', 'trainIdx', 'queryIdx', 'imgIdx'])
        writer.writerow(['example = ', str(match[0])])
        for i in range(0, len(match) - 1):
            writer.writerow([str(2 * i + 1),
                             str(type(match[i][0])),
                             str(match[i][0].distance),
                             str(match[i][0].trainIdx),
                             str(match[i][0].queryIdx),
                             str(0)])  # str(match[i][0].imgIdx)])
            writer.writerow([str(2 * i + 2),
                             str(type(match[i][1])),
                             str(match[i][1].distance),
                             str(match[i][1].trainIdx),
                             str(match[i][1].queryIdx),
                             str(1)])  # str(match[i][1].imgIdx)])
    print(match_name + 'is updated')


def showimg(framesize, img_name, img):
    img = cv2.resize(img, (256 * framesize, 256 * framesize))
    cv2.imshow(img_name, img)


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


def BFmatch(ori_p, noise_p, distance):
    '''
    knnmatch匹配的参数k，表示每个关键点保留的最佳即最短距离匹配的最大数量。下例取k=2，即每个查询关键点有两个最佳匹配。

    次近邻的距离乘以一个小于1的值，获得的值称为阈值，只有当距离分值小于阈值时，则称之为“好点”。最近邻和次近邻大小差不多的点，更容易出现错误的匹配，即“坏点”。这种方法时比率检验。

    应用比率检验，下例代码将阈值设置为次优匹配距离分值的0.75倍，如果knnmatch不满足次优匹配，则该点为“坏点”，舍弃。
    '''
    center_ori = []
    center_noise = []
    ori_img = cv2.imread(ori_p)
    # ori_img = cv2.resize(ori_i, (256 * 3, 256 * 3))
    noise_img = cv2.imread(noise_p)
    # noise_img = cv2.resize(noise_i, (256 * 3, 256 * 3))
    # hessian = 100
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.03,  # 去除对比度低的点  DOG空间极值检测
        edgeThreshold=10,  # 去除不稳定的边缘响应点
        sigma=1.6
    )
    keypoints_ori, descriptor_ori = sift.detectAndCompute(ori_img, None)
    keypoints_noise, descriptor_noise = sift.detectAndCompute(noise_img, None)
    # kp_second_ori = []
    # dt_second_o = []
    # kp_second_noise = []
    # dt_second_n = []
    # for i in range(0, len(keypoints_ori)):
    #     if unpackOctave(keypoints_ori[i]) == 2:
    #         kp_second_ori.append(keypoints_ori[i])
    #         dt_second_o.append(descriptor_ori[i])
    # for j in range(0, len(keypoints_noise)):
    #     if unpackOctave(keypoints_noise[j]) == 2:
    #         kp_second_noise.append(keypoints_noise[j])
    #         dt_second_n.append(descriptor_noise[j])
    # dt_second_ori = np.array(dt_second_o)
    # dt_second_noise = np.array(dt_second_n)
    #
    # print(len(keypoints_ori))
    # print(descriptor_ori)
    # print(len(descriptor_ori))
    # print(dt_second_ori)
    # print(len(dt_second_ori))

    kp_oriimg = cv2.drawKeypoints(ori_img, keypoints_ori, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                  (51, 163, 236), 4)
    writekp(keypoints_ori, ori_kp_name)
    kp_noiseimg = cv2.drawKeypoints(noise_img, keypoints_noise, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                    (255, 192, 203), 4)
    if len(keypoints_noise) == 0:
        print('fail = 0')
    else:
        print('true = ' + str(len(keypoints_noise)))
        writekp(keypoints_noise, noise_kp_name)
        bf = cv2.BFMatcher()

        res = bf.match(descriptor_ori, descriptor_noise)
        res = sorted(res, key=lambda x: x.distance)
        # print('resresresresresresresresresresresresresresresresresresresres')
        # print(res)
        with open(matchname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['num', 'type', 'distance', 'trainIdx', 'queryIdx', 'imgIdx'])
            writer.writerow(['example = ', str(res[0])])
            for i in range(0, len(res) - 1):
                writer.writerow([str(i + 1),
                                 str(type(res[i])),
                                 str(res[i].distance),
                                 str(res[i].trainIdx),
                                 str(res[i].queryIdx),
                                 str(res[i].imgIdx)])
        print(matchname + ' is updated')

        with open(good_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['num', 'type', 'distance', 'trainIdx', 'queryIdx', 'imgIdx'])
            good = []  # 这里主要是匹配的更加精确，可以修改ratio值得到不同的效果，即下面的0.75
            trainindex = []
            queryindex = []
            for m in res:
                if m.distance < distance:
                    good.append([m])
                    writer.writerow([str(i),
                                     str(type(m)),
                                     str(m.distance),
                                     str(m.trainIdx),  # 右边那张小图
                                     str(m.queryIdx),  # 左边那张大图
                                     str(m.imgIdx)])
                    i = i + 1
                    trainindex.append(m.trainIdx)
                    queryindex.append(m.queryIdx)
            print('good.csv is updated')

        match = re.search(r'\d+(?=\.png)', noise_p)
        number = match.group()
        print('第' + str(number) + '张图片匹配')
        number = int(number)
        if (number % 12) == 0:
            xmin = 176
            ymin = int(number / 12 - 1) * 16  # 0-198
        else:
            xmin = (number % 12 - 1) * 16  # 0-198
            ymin = int(number / 12) * 16  # 0-198
        print('xmin = ' + str(xmin))
        print('ymin = ' + str(ymin))

        good_num = len(trainindex)
        print('足够健壮的sift匹配对个数 = ' + str(good_num))
        center_ori = np.array(get_rect(res, keypoints_ori, good_num, 0))

        center_noise = np.array(get_rect(res, keypoints_noise, good_num, 1))
        print(center_ori)
        # print(get_rect(res, keypoints_ori, good_num, 0))
        print(center_noise)

        N = len(center_ori)  # 一共多少匹配对
        c = []
        for i in range (N):
            c.append([[elem[0] - xmin, elem[1] - ymin] for elem in center_ori[i]])
        center = np.array(c)

        print(center)

        # 计算arr1和arr2中所有元素的两两距离
        distances = np.sqrt(np.sum((center[:, np.newaxis, :] - center_noise) ** 2, axis=-1))
        print(distances)

        # 找出最小距离对应的索引
        min_index = np.unravel_index(np.argmin(distances), distances.shape)

        # 最小距离
        min_distance = distances[min_index]

        print(min_index, min_distance)

        N1 = 0
        if len(distances) != 0:
            for i in range(len(distances)):
                for j in range(len(distances[i])):
                    if distances[i][j] < np.sqrt(18):
                        N1 += 1
        print(N1)

        print(N)


        if min_distance > np.sqrt(18):
            print("匹配失败")  # 表明没有一个特征点对是符合要求的
        elif min_distance < np.sqrt(18) and N1/N > 0.3:
            print("匹配成功")
            print('健壮匹配对有 ' + str(N1) + ' 对')
        else:
            print("匹配失败")
            print('健壮匹配对有 ' + str(N1) + ' 对')


        # center = []
        # for i in range(len(center_ori)):
        #     for j in range(0,2):
        #         center_ori[i][j] -
        #     center
        #
        # a= center_ori[1] - xmin
        # print(a)




        # print(min3)
        # print(max4)
        # cv2.rectangle(ori_img, min1, max2, [255, 255, 255], 4, 16)
        # cv2.rectangle(noise_img, min3, max4, [255, 255, 255], 4, 16)

        # cv2.rectangle(ori_img, min1, max2, [255, 255, 255], 2)
        # cv2.rectangle(noise_img, min3, max4, [255, 255, 255], 2)

        # matches = bf.knnMatch(descriptor_ori, descriptor_noise, k=2)
        # print(matches)
        # writematch(matches, matches_name)
        # print(matches)
        # with open(good_name, 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['num', 'type', 'distance', 'trainIdx', 'queryIdx', 'imgIdx'])
        #     i = 1
        #     good = []  # 这里主要是匹配的更加精确，可以修改ratio值得到不同的效果，即下面的0.75
        #     trainindex = []
        #     queryindex = []
        #
        #     for m, n in matches:
        #         # matches = sorted(matches, key=lambda x: x.distance)
        #         if m.distance < 0.7 * n.distance and m.distance < 300:
        #             # res
        #             good.append([m])
        #             # print([m])
        #             writer.writerow([str(i),
        #                              str(type(m)),
        #                              str(m.distance),
        #                              str(m.trainIdx),  # 右边那张小图
        #                              str(m.queryIdx),  # 左边那张大图
        #                              str(m.imgIdx)])
        #             i = i + 1
        #             trainindex.append(m.trainIdx)
        #             queryindex.append(m.queryIdx)
        #     print(good)
        # good.sort(key=lambda x: x.distance)

        # '''获得优秀sift对应具体坐标'''
        # print(trainindex)  # 右边那张小图
        # with open(noise_kp_name, 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     for i, rows in enumerate(reader):
        #         for r in range(0, len(trainindex)):
        #             q = int(trainindex[r]) * 2 + 2
        #             if i == q:
        #                 row = rows
        #                 # print(row)
        #                 print(row[2])  # 获得noisekp坐标
        #
        # print(queryindex)  # 左边那张大图
        # with open(ori_kp_name, 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     for i, rows in enumerate(reader):
        #         for r in range(0, len(queryindex)):
        #             q = int(queryindex[r]) * 2 + 2
        #             if i == q:
        #                 row = rows
        #                 # print(row)
        #                 print(row[2])  # 获得orikp坐标

        # '''对于一个特征点会不会对应另一张图上多个特征点，如果出现这样的情况那么说明匹配失败，这是第一步一对多的判断'''
        # for i in range(0, len(trainindex) - 1):
        #     if trainindex[i] == trainindex[i + 1]:
        #         print('bad sift(same)')
        #         break
        #     # else:
        #     #     print('good sift(different)')
        #
        #
        # if within_range(min1[0], xmin, min1[1], ymin):
        #     print(f"匹配成功")
        # else:
        #     print(f"匹配失败")

        result_img = cv2.drawMatchesKnn(ori_img, keypoints_ori, noise_img, keypoints_noise, good, None,
                                        flags=2)
        # print(good)
        # print(len(good))

        cv2.imshow("match result", result_img)



def within_range(min1x, xmin, min1y, ymin):
    if abs(min1x - xmin) <= 3 and abs(min1y - ymin) <= 3:
        return True
    else:
        return False


def get_rect(res, kp, goodnum, index):
    point_img = []
    center = []

    # dt_second_ori = np.array(dt_second_o)
    for i in res[:goodnum]:
        if index == 0:
            center.append(cv2.KeyPoint_convert(kp, keypointIndexes=[i.queryIdx]))
            # print(center_ori)
        elif index == 1:
            center.append(cv2.KeyPoint_convert(kp, keypointIndexes=[i.trainIdx]))
            # print(center_noise)

    # print(center_ori)
    # print(center_noise)

    # for j in range(0, len(center_ori)):

        # center = [int(np.ravel(center_ori)[0]), int(np.ravel(center_noise)[1])]

        # point_img.append(center)
    # minres = np.argmin(point_img, axis=0)
    # maxres = np.argmax(point_img, axis=0)
    # minpoint = [point_img[minres[0]][0], point_img[minres[1]][1]]
    # maxpoint = [point_img[maxres[0]][0], point_img[maxres[1]][1]]
    return center


def FLANNmatch(ori_p, noise_p):
    '''
    knnmatch匹配的参数k，表示每个关键点保留的最佳即最短距离匹配的最大数量。下例取k=2，即每个查询关键点有两个最佳匹配。

    次近邻的距离乘以一个小于1的值，获得的值称为阈值，只有当距离分值小于阈值时，则称之为“好点”。最近邻和次近邻大小差不多的点，更容易出现错误的匹配，即“坏点”。这种方法时比率检验。

    应用比率检验，下例代码将阈值设置为次优匹配距离分值的0.75倍，如果knnmatch不满足次优匹配，则该点为“坏点”，舍弃。
    '''
    ori_img = cv2.imread(ori_p)
    cv2.resize(ori_img, (256 * 3, 256 * 3))
    noise_img = cv2.imread(noise_p)
    cv2.resize(noise_img, (256 * 3, 256 * 3))
    hessian = 100
    sift = cv2.SIFT_create(hessian)
    keypoints_ori, descriptor_ori = sift.detectAndCompute(ori_img, None)
    keypoints_noise, descriptor_noise = sift.detectAndCompute(noise_img, None)
    kp_oriimg = cv2.drawKeypoints(ori_img, keypoints_ori, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, (51, 163, 236), 4)
    writekp(keypoints_ori, ori_kp_name)
    kp_noiseimg = cv2.drawKeypoints(noise_img, keypoints_noise, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                    (255, 192, 203), 4)
    writekp(keypoints_noise, noise_kp_name)

    # 设置参数
    FLANN_INDEX_KDTREE = 1  # 建立FLANN匹配器的参数
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=500)  # 递归次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_ori, descriptor_noise, k=2)

    # print(matches)
    writematch(matches, matches_name)
    # print(matches)
    with open(good_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num', 'type', 'distance', 'trainIdx', 'queryIdx', 'imgIdx'])
        i = 1
        good = []  # 这里主要是匹配的更加精确，可以修改ratio值得到不同的效果，即下面的0.75
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                # if m.distance < 240:
                good.append([m])
                # print([m])
                writer.writerow([str(i),
                                 str(type(m)),
                                 str(m.distance),
                                 str(m.trainIdx),
                                 str(m.queryIdx),
                                 str(m.imgIdx)])

                i = i + 1
    result_img = cv2.drawMatchesKnn(ori_img, keypoints_ori, noise_img, keypoints_noise, good, None, flags=2)
    print(good)
    # writegood(good, good_name)
    print(len(good))

    cv2.imshow("match result", result_img)


if __name__ == '__main__':
    noise1_path = r'./ultimatenoise_G=0.1M=0.4.png'  # highest noise
    noise2_path = r'./ultimatenoise_G=0.05M=0.2.png'  # higher
    noise3_path = r'./ultimatenoise_G=0.02M=0.1.png'  # medium
    cut_noise3_path = r'./cut/cut1_5.png'  # medium
    noise4_path = r'./ultimatenoise_G=0.01M=0.04.png'  # lower
    noise5_path = r'./ultimatenoise_G=0.0005M=0.02.png'  # lowest
    ori_kp_name = r'orikeypoints.csv'
    noise_kp_name = r'noisekeypoints.csv'
    matches_name = r'matches.csv'
    matchname = r'match.csv'
    good_name = r'good.csv'

    ori_path = r'./originopt/opt1.png'
    cut_path = r'./cut/cut54/cut54_132.png'
    BFmatch(ori_path, cut_path, 180)
    # test_path = r'./sarcut/sarcutcut1_1.png'
    # BFmatch(ori_path, test_path, 280)
    # judge_same(good_name)
    # BFmatch(ori_path, noise3_path)
    # FLANNmatch(ori_path, noise5_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
