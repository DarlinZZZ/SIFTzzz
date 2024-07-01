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
def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / (float)(1 << octave) if octave >= 0 else (float)(1 << -octave)
    return octave, layer, scale


def test():
    ori_img = cv2.imread(r'./opt1.png')
    ortimg = cv2.resize(ori_img, (256 * 3, 256 * 3))
    noise_img = cv2.imread(r'./ultimatenoise_G=0.0005M=0.02.png')
    cv2.resize(noise_img, (256 * 3, 256 * 3))
    hessian = 100  # 取100个初始sift特征点

    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.03,
        edgeThreshold=10,
        sigma=1.2
    )
    keypoints, descriptor = sift.detectAndCompute(ortimg, None)
    array = []
    for values in unpackOctave(keypoints[0]):
        array.append(values)
    print(array)
    # unpackOctave(keypoints)
    print(keypoints)
    kp_second = []
    for k in keypoints:
        if k.octave >10000000:
            kp_second.append(k)
    print(kp_second)
    out_img = cv2.drawKeypoints(ortimg, kp_second, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, (51, 163, 236), 4)
    # img = cv2.resize(out_img, (256 * 3, 256 * 3))
    cv2.imshow('SIFT', out_img)


if __name__ == '__main__':
    test()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
