import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
#======================
# 用于读取高分辨率数据集
#======================
#数据预处理，把原图像处理成小图和大图
class DataLoader():
    #初始化，重构后清晰图像的大小
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
    #从文件夹里读数据
    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/train/*' % (self.dataset_name))

        #随机选图片训练，可能很多张
        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            #计算缩小的数据
            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            #将图片缩小
            img_hr = cv2.resize(img, self.img_res)
            img_lr = cv2.resize(img, (low_h, low_w))

            # If training => do random flip，如果是训练模式，翻转，做数据增强
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        #归一化 0-255，255/127.5=2，0-2之间，-1就归一化到-1到1之间
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr #矩阵，列表里放的矩阵

    #读图片，转化到RGB
    def imread(self, path):
         img =cv2.imread(path)
         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)