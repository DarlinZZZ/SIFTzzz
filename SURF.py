'''SURF加速SIFT算法'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = r'./opt1.png'
img = cv2.imread(img_path)
# 参数为hessian矩阵的阈值
surf = cv2.SIFT_create(400)
# 设置是否要检测方向
# surf.setUpright(True)
# 输出设置值
#print(surf.getUpright())
# 找到关键点和描述符
kp, des = surf.detectAndCompute(img, None)
# 把特征点标记到图片上
img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
# 输出描述符的个数
print(surf.descriptorSize())
plt.imshow(img2), plt.show()
