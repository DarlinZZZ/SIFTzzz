import numpy as np


def number():
    number = 96
    # 256 = 64 +16*12
    if (number % 12) == 0:
        xmin = 176
        ymin = int(number / 12 - 1) * 16  # 0-198
    else:
        xmin = (number % 12 - 1) * 16  # 0-198
        ymin = int(number / 12) * 16  # 0-198
    print(xmin)
    print(ymin)


def MSE():
    center = []
    center_noise = []

    center.append([29.58952332, 30.89466858-4])
    center.append([29.58952332 - 2, 30.89466858 - 3])
    # center.append([30.130081, 30.716663])
    center_noise.append([29.58952332, 30.89466858])
    center_noise.append([19.58952332, 10.89466858])
    c = np.array(center)
    cn = np.array(center_noise)

    # [[[30.130081 30.716663]]
    # [[30.020344 26.037758]]
    # [[30.020344 26.037758]]
    # [[30.020344 26.037758]]]
    #
    # [[[29.58952332  30.89466858]]
    # [[24.49739075 - 5.91165161]]
    # [[101.87409973  12.61334229]]
    # [[101.87409973  12.61334229]]]

    # 计算arr1和arr2中所有元素的两两距离
    distances = np.sqrt(np.sum((c[:, np.newaxis, :] - cn) ** 2, axis=-1))
    print(distances)
    # 找出最小距离对应的索引
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    # 最小距离
    min_distance = distances[min_index]

    print(min_index, min_distance)

    N1 = 0
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if distances[i][j] < np.sqrt(18):
                N1 += 1

    print(N1)
    '''
    首先计算两个数组中所有元素的欧式距离，得到一个距离矩阵distances。其中，np.newaxis用于在arr1中增加一个维度，以便与arr2进行广播计算；
    找出距离矩阵中最小距离对应的索引，使用np.unravel_index函数将最小元素的位置转换为对应的行列坐标；
    最小距离即为距离矩阵中最小元素的值。
    注意，上面的代码中使用了numpy的广播机制，可以避免使用循环，提高计算效率。
    '''


if __name__ == '__main__':
    MSE()
