from PIL import Image
import os
'''
cut_image(path, savepath, number)
功能：切割可见光图片，切割数量为number
256*256 x,y平均等分成12*12个子图,step = (256-64)/(number**0.5)
'''
def cut_image(path, save_path, savepath, number, imgnum):
    # img_path = r'./opt1.png'
    # img_savepath = r'./cut/cut1.png'
    img = Image.open(path)
    step = (256-64)/(number**0.5)
    count = 1
    # x = []
    # y = []
    for i in range(1, 13):
        for j in range(1, 13):
            xmin = step * (j - 1)
            ymin = step * (i - 1)
            xmax = 64 + step * (j - 1)
            ymax = 64 + step * (i - 1)
            temp = img.crop((xmin, ymin, xmax, ymax))
            temp.save(save_path + r'cut' + str(imgnum) + '_' + str(count) + '.png')
            # print('xmin = ' + str(xmin))
            # print('ymin = ' + str(ymin))
            # x.append(xmin)
            # y.append(ymin)
            count += 1
    # print(x)
    # print('cut image saved')
    # return x,y


def cuts(path, savepath, number):
    for i in range (1, 1697):
        imgpath = path + str(i) + '.png'
        save_path = r'./cut/cut'+str(i)+'/'
        cut_image(imgpath, save_path, savepath, number, i)
        print('cut' + str(i)+' saved')
    print('cut finished')

def test_cuts(path, savepath, number):
    imgpath = path
    save_path = r'./sarcut'
    cut_image(imgpath, save_path, savepath, number, 1)
    print('cut finished')


def mkfile():
    for i in range (1,1697):
        os.mkdir("./cut/cut{}".format(i))


if __name__ == '__main__':
    # img_path = r'./opt1.png'
    # x = []
    # y = []
    noise1_path = r'./ultimatenoise_G=0.1M=0.4.png'  # highest noise
    noise2_path = r'./ultimatenoise_G=0.05M=0.2.png'  # higher
    noise3_path = r'./ultimatenoise_G=0.02M=0.1.png'  # medium
    noise4_path = r'./ultimatenoise_G=0.01M=0.04.png'  # lower
    noise5_path = r'./ultimatenoise_G=0.005M=0.02.png'  # lowerr
    noise6_path = r'./ultimatenoise_G=0.0005M=0.02.png'  # lowest
    noise_path = r'./noise/noise'
    test_path = r'./sar1.png'
    test_savepath = r'sar1.png'
    img_savepath = r'cut1.png'
    # mkfile()
    test_cuts(test_path, test_savepath, 144)
    # cuts(noise_path, img_savepath, 144)  # 切割小图片

