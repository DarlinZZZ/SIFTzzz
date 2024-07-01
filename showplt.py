import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import random
import csv


def read(csvpath):  # 读取csv文件
    with open(csvpath, 'r') as file:
        reader = csv.reader(file)
        numbers = []

        for i, rows in enumerate(reader):
            next(reader)
            # if i >= 1 and i <= 1600:
            if i >= 1 and i <= 401:
                row = float(rows[1])
                numbers.append(row * 100)
        number = np.array(numbers)

    return number


def Cumulative_Histogram(data):
    '''
    绘制曲线累计图
    '''
    hist, bins = np.histogram(data, bins=range(101))
    cumulative_hist = np.cumsum(hist)
    # 绘制累计直方图
    plt.plot(bins[:-1], cumulative_hist)
    plt.xlabel("ratio(%)")
    plt.ylabel("Cumulative Count")
    plt.title("Cumulative Histogram")  # 累计直方图
    plt.show()


def hist(data):
    '''
    累计直方图
    '''
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=1600, cumulative=True, density=True, weights=None)
    ax.set_title('Cumulative Histogram')
    ax.set_xlabel('ratio(%)')
    ax.set_ylabel('Cumulative Frequency')
    plt.show()


def ratio_1600(data):
    '''
    1600张图片各自匹配概率
    '''
    fig1 = plt.figure("ratio_1600", figsize=(16, 9))
    fig1.canvas.manager.window.wm_geometry('+0+0')  # 左上角显示
    plt.hist(data, bins=1600, range=(0, 100))

    # 设置x轴和y轴标签
    plt.xlabel("ratio(%)")
    plt.ylabel("total img number in certain ratio")

    # 显示图形
    plt.show()


def ratio_5(number):
    '''
    可显示匹配概率 >80% 的图片数量（总量=1600）
    '''
    y = number
    x = []
    for i in range(0, len(y)):
        x.append(i)
    fig1 = plt.figure("ratio_5", figsize=(10, 6))
    fig1.canvas.manager.window.wm_geometry('+0+0')  # 左上角显示
    n, bins, patches = plt.hist(y, bins=5, rwidth=0.97)

    for i in range(0, len(n)):
        # plt.text(bins[i], n[i] * 1.02, 'ratio>' + str(bins[i])+'<'+ str(bins[i+1]) + '=' + str(int(n[i])), fontsize=9,
        #          horizontalalignment="left")  # 打标签，在合适的位置标注每个直方图上面样本数
        plt.text(bins[i], n[i] * 1.02,
                 'ratio=' + str(int(n[i])) + ' Prop=' + str(round(int(n[i])/16,4)) + '%',
                 fontsize=9,
                 horizontalalignment="left")
    plt.legend()
    plt.xlabel("ratio(%)")
    plt.ylabel("total img number in certain ratio")
    plt.show()


if __name__ == '__main__':
    # csv_path = r'./result.csv'
    csv_path = r'./result_sorted.csv'
    # Cumulative_Histogram(read(csv_path))
    # hist(read(csv_path))
    # ratio_1600(read(csv_path))
    ratio_5(read(csv_path))
