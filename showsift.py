import csvfile as csvfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import getcwd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv

'''
SIFT特征的生成一般包括以下几个步骤：
（1） 构建尺度空间，检测极值点，获得尺度不变性。
（2） 特征点过滤并进行精确定位。
（3） 为特征点分配方向值。
（4） 生成特征描述子。
可能从关键点周围的3×3区域的强度值形成一个向量，具有旋转不变性
'''


def showimg(framesize, img_name, img):
    img = cv2.resize(img, (256 * framesize, 256 * framesize))
    cv2.imshow(img_name, img)


'''
cv2.DRAW_MATCHES_FLAGS_DEFAULT：创建输出图像矩阵，使用现存的输出图像绘制匹配对和特征点，对每一个关键点只绘制中间点
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：不创建输出图像矩阵，而是在输出图像上绘制匹配对
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：对每一个特征点绘制带大小和方向的关键点图形
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
'''


def writekp(keypoints, kp_name):
    '''
    功能：将关键点信息写入csv文件中
    :param keypoints: 关键点
    :param kp_name: csv文件名
    pt:坐标信息
    size:点领域大小（直径大小）
    angle:特征点方向
    response:特征点响应程度指数(越大，关键点越好），解释为该特征实际存在的概率
    octave:特征点所在的金字塔组（从哪一层得到的数据）
    class_id:类型（可以进行分类，默认为-1）
    '''
    with open(kp_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num', 'type', 'position', 'size', 'angle', 'response', 'octave', 'class_id'])
        writer.writerow(['example = ', str(keypoints[1])])
        for i in range(0, len(keypoints)):
            writer.writerow([str(i),
                             str(type(keypoints[i])),
                             str(keypoints[i].pt),
                             str(keypoints[i].size),
                             str(keypoints[i].angle),
                             str(keypoints[i].response),
                             str(unpackOctave(keypoints[i])),
                             str(keypoints[i].class_id)])
    print('keypoints.csv is updated')

def writedt(descriptor, dt_name_beforepca, dt_name_pca):
    '''
    为什么？隐含了一个假设，就是训练数据和测试数据实际上是同分布的（因此我们才可以使用训练数据集来预测测试数据集），来自于同一个总体。
    StandardScaler类是一个用来讲数据进行归一化和标准化的类。
    结果：对于每个属性/每列来说所有数据都聚集在0附近，标准差为1，使得新的X数据集方差为1，均值为0
    fit_transform()方法：是fit和transform的结合，意为找出均值和标准差并且通过居中和缩放执行标准化
    :return 将detectAndCompute()得到的100个128维描述符集标准化处理，使得方差为1，均值为0
    '''
    descriptor = StandardScaler().fit_transform(descriptor)
    with open(dt_name_beforepca, 'w') as csvfile:
        writer = csv.writer(csvfile)
        list = [[j] for j in range(1,len(descriptor)+1)]
        dd = np.hstack((list, descriptor))  # 编序号
        for i in range(0, len(dd)):
            writer.writerow(dd[i])
    print('dt_name_beforepca.csv is updated')
    '''
    pca的一般步骤：先对原始数据零均值化，然后求协方差矩阵，接着对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。（无监督学习算法）
    n_components: PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
    pca.fit用decriptor训练pca模型本身，并且返回降度后的数据。
    '''
    pca = PCA(n_components="mle")
    pcadescriptor = pca.fit_transform(descriptor)
    with open(dt_name_pca, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(0, len(pcadescriptor)):
            writer.writerow([str(i+1),pcadescriptor[i]])
    print("-------------------------------------------------")
    print(pca.singular_values_)  # 查看特征值
    print("-------------------------------------------------")
    print(pca.components_)  # 打印查看特征值对应的特征向量
    with open(dt_name_components, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(0, len(pca.components_)):
            writer.writerow(pca.components_[i])
    print('dt_name_beforepca.csv is updated')
    print("-------------------------------------------------")
    print(pcadescriptor)
    print("-------------------------------------------------")
    '''
    explained_variance_ratio_:查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比，又叫做可解释方差贡献率。
    '''
    print(pca.explained_variance_ratio_)
    # print('descriptor.txt is updated')
    print("-------------------------------------------------")
    inverse_descriptor = pca.inverse_transform(pcadescriptor)
    print(inverse_descriptor)
    print("-------------------------------------------------")
    # img = cv2.resize(img, (256 * framesize, 256 * framesize))
    # cv2.imshow(img_name, img)
    # for j in range(len(inverse_descriptor)):
    #     plt.imshow(inverse_descriptor[j].reshape(128,1), cmap="binary_r")



'''
SIFT(输入图片路径， SIFT特征数量， 显示边框大小256*framesize， 颜色012）
cv2.SIFT_create(参数min_hessian 阈值)
值=特征点个数
detectAndCompute方法主要执行两项操作：特征检测和描述符计算。该操作的返回值是一个元组，包含一个关键点列表和另一个关键点的描述符列表。
detectAndCompute详情：https://blog.csdn.net/xdEddy/article/details/78206459?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-78206459-blog-107850859.235%5Ev28%5Epc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-78206459-blog-107850859.235%5Ev28%5Epc_relevant_default&utm_relevant_index=9
定义了一个指针，这个指针由一个静态全局数组生成bit_pattern_31_，直接使用随机的pattern来算描述子，makeRandomPattern用了一个固定的随机数种子，肯定能得到相同的pattern。
如果wta_k为2，那直接就用bit_pattern_31_里的数据来取pattern。否则调用initializeOrbPattern在bit_pattern_31_里随机取点来组成pattern
其中根据keypoints[j]的angel算了个cos即a和sinb以备后用。显然，算的描述子是要支持旋转的。
到这里也就能看出pattern里到底存的啥了，其实就是一堆坐标。用的时候把坐标旋转到灰度质心法给出的角上以实现旋转不变性的描述子特性。
wta_k=2的情况。算法是8对8对的比较大小的。for循环里每次比较8对，形成一个字节的描述子，这样运行dsize=32次就形成了终极目标：256位的描述子。
8对点即16个点，也就是在pattern上取16个点。
'''
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
#
def SIFT(path, sift_number, framesize, c):
    img = cv2.imread(r'./opt1.png')
    img = cv2.resize(img, (256 * framesize, 256 * framesize))
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=5,
        contrastThreshold=0.03,  # 去除对比度低的点  DOG空间极值检测
        edgeThreshold=10,  # 去除不稳定的边缘响应点
        sigma=1.6
    )
    kp_second = []
    keypoints, descriptor = sift.detectAndCompute(img, None)
    for i in range(0, len(keypoints)):
        if unpackOctave(keypoints[i]) != 5:
            kp_second.append(keypoints[i])
            # del keypoints[i]
            # keypoints.del(keypoints[i])
    print(kp_second)
    print(len(keypoints))
    print(len(kp_second))


    # kp_second = []
    # for k in keypoints:
    #     if k.octave == 2:
    #         kp_second.append(k)
    out_img = cv2.drawKeypoints(img, tuple(kp_second), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color[c], 4)
    # writekp(tuple(kp_second), kp_name)
    # out_img = cv2.drawKeypoints(img, keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color[c], 4)
    writekp(keypoints, kp_name)
    # writedt(descriptor, dt_name_before, dt_name_pca)
    showimg(3, "SIFT", out_img)


if __name__ == '__main__':
    img_path = r'./interference/grayimg.png'
    kp_name = 'keypoints.csv'
    dt_name_before = 'descriptor_beforepca.csv'
    dt_name_pca = 'descriptor_afterpca.csv'
    dt_name_components = 'descriptor_components.csv'
    color = [(51, 163, 236), (255, 0, 0), (255, 192, 203)]  # 0-orange ; 1-blue; 2-purple
    SIFT(img_path, 200, 3, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()















