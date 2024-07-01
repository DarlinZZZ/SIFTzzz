import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import random
import csv
import os
import shutil


def read(csvpath):  # 读取csv文件
    with open(csvpath, 'r') as file:
        reader = csv.reader(file)
        numbers = []

        for i, rows in enumerate(reader):
            next(reader)
            # if i >= 1 and i <= 1600:
            if i >= 1 and i <= 401:
                row = int(rows[0])
                numbers.append(row)
        number = np.array(numbers)
    # print(number)
    print(len(number))
    return number



def ctrlx(number):
    '''
    将匹配概率大于75的图片分成第一类放入文件夹：1  把剩余的放入文件夹：2
    :param number:  401张 匹配概率大于75%的图片号码
    :return:
    '''
    # 数字所在的文件夹位置
    path_2 = "./2"

    # 数字照片剪切到的目标文件夹位置
    path_1 = "./1"

    # 遍历数组中的每个数字
    for num in number:

        # 遍历文件夹中的所有文件
        for filename in os.listdir(path_2):

            if filename.endswith('opt' + str(num) + ".png"):
                shutil.move(path_2+'/'+filename, path_1+'/'+filename)
    print('ctrlx finished')




if __name__ == '__main__':
    csv_path = r'./result_sorted.csv'
    ctrlx(read(csv_path))
