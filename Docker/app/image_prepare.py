import os
import random
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from dataset import Dataset
import tensorflow as tf

# import random
# random.seed(1)

# DATA_ROOT = '../pages/'

# class_names = {'Void':0,'DLSignature':1,'DLLogo':2}

# # target_img_size = (1200,1600)
# target_img_size = (600,800)
# win_size = 224

class ImagePrepare():
    def __init__(self, win_size, target_img_size):
        self.win_size =  win_size
        self.target_img_size = target_img_size

        # Для того чтобы загнать катринку в нейронную сеть, необходимо разбить изображение на квадраты размером win_size.
        # следующие два параметра оптределяют к-во этих квадратов. 
        self.steps_x = math.ceil(self.target_img_size[0]/self.win_size)
        self.steps_y = math.ceil(self.target_img_size[1]/self.win_size)

        # запомнить позицию каждого квадрата.
        self.step_x = math.floor(self.target_img_size[0]/math.ceil(self.target_img_size[0]/self.win_size)) 
        self.step_y = math.floor(self.target_img_size[1]/math.ceil(self.target_img_size[1]/self.win_size))
        self.windows_coords = self.calc_window_coords


# Необходимо принять изображение. Определить его исходный размер. 
# уменьшить его размер до 600х800.
# Разрезать на составные части
# Получить ответ.
# Объединить прямоугольники, определить их реальные координаты. 
# Пересичиатьих позицию на исходном изображении.
# Вывести ответ в виде словаря.

# def read_image(path):
#     return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BAYER_BG2GRAY)
#==========================================================================

    def create_mask(self, bb, x, class_index):
        """Создаем маску для bounding box'a такого же шейпа как и изображение"""
        rows,cols, _ = x.shape
        y_ind = np.zeros((rows, cols)) # Нужно больше каналов.
        # y_cat = np.zeros((rows, cols , len(class_names)-1))
        bb = bb.astype(np.int)

        if class_index > 0:
            y_ind[bb[1]:bb[3]+bb[1],bb[0]: bb[2]+bb[0]] = class_index # У нас два класса. И, для каждого из них нужно сделать маску.
            # y_cat[bb[1]:bb[3]+bb[1],bb[0]: bb[2]+bb[0], class_index-1] = 1
        return y_ind

    def create_bb_array(self, x,class_index):
        """Генерируем массив bounding box'a из столбца train_df"""
        if class_index == 0:
            res = np.array([0,0,0,0]) 
        else:
            res = np.array([x[0],x[1],x[2],x[3]])      
        return res

    #=========================================================================
    def mask_to_bb(self, mask):
        """Конвертируем маску Y в bounding box'a, принимая 0 как фоновый ненулевой объект """
        # bbx = []
        # for i in range(len(class_names)-1):
        cols, rows = np.nonzero(mask)
        if len(cols) == 0:
            bb=[-1,-1,-1,-1]
        else:
            top_row = np.min(rows)
            left_col = np.min(cols)
            bottom_row = np.max(rows)
            right_col = np.max(cols)
            bb = [top_row, left_col, bottom_row-top_row, right_col-left_col]
        return np.array(bb, dtype=np.float32)

    #========================================================================================
    def resize_image_bb(self, img_size,img_new_size, bb):
        width, height =  img_size[0], img_size[1]
        new_width, new_height  = img_new_size[0], img_new_size[1]
        height_scale = new_height/height
        width_scale = new_width/width
        bb[0] *= width_scale 
        bb[1] *= height_scale
        bb[2] *= width_scale
        bb[3] *= height_scale
        return bb 

  # Исходное изображение имеет очень большой размер. т.о. оно не помещается в нейронную сеть. Так что оно помещается в нейронную сеть целиком. Для того чтобы обойти это ограничение разрежем изображение на квадраты размером `win_size` c небольшим взаимным перекрытием. И в таком виде будем подавать модель в сеть.
  # Вычисляем границы 
    def calc_window_coords(self):
        win_coords = [ ]
        for i in range(self.steps_y):
            for j in range(self.steps_x):
                pos_x = self.step_x*j
                pos_y = self.step_y*i
                if pos_x+self.win_size> self.target_img_size[0]:
                    pos_x = self.target_img_size[0]-self.win_size
                if pos_y+self.win_size> self.target_img_size[1]:
                    pos_y = self.target_img_size[1]-self.win_size   
                win_coords.append([pos_y, pos_x, self.win_size, self.win_size])
        return win_coords

    # Процедура для разбиения изображения на квадраты
    def split_image(self, img, bbx=None, class_index=1):
        is_not_bbx =False
        if not bbx:
            is_not_bbx=True
            bbx = np.array([-1,-1,-1,-1], dtype='float')
        mask = self.create_mask(bbx, img, class_index)
    # разделем изображение на составные части и возращем кусочки small_imgs and small_bbx
        img_combined = np.concatenate([img, mask[..., None]], axis=2)
        X_batch=[]
        y_batch=[]
        for coord in self.win_coords:
            combined = img_combined[coord[0]:(coord[0]+coord[2]), coord[1]:(coord[1]+coord[3])]
            X_batch.append(combined[...,:3])
            bb = self.mask_to_bb(combined[...,3])
            y_batch.append(bb)

        X_batch = np.array(X_batch, dtype='float') /255.0
        y_batch = np.array(y_batch, dtype='float') /255.0

        if is_not_bbx:
            return X_batch
        else:    
            return X_batch, y_batch

    # def show_result(self, sample_imgs,sample_bbx):
    #     fig = plt.figure(figsize=(15, 15))
    #     for j in range(len(sample_imgs)):
    #         ax = fig.add_subplot(steps_y, steps_x, j+1)
    #         ax.imshow(sample_imgs[j])
    #         bb = sample_bbx[j]
    #         rect = plt.Rectangle((bb[0]*win_size, bb[1]*win_size), bb[2]*win_size, bb[3]*win_size,fill=False, color='red')
    #         ax.add_patch(rect)
    #         plt.xticks([]), plt.yticks([])
    #     plt.show()


    # определим реальную позицию прямоугольников и их координаты на изначальнои изображении.
    # А дальше по всем точкм пройдемся. 
    # Еще нужно будет отсечь наименее вероятные. Но, это позже.

    # win_coords
    def combined_bb(self, bbx):
        x,y,x1,y1= self.target_img_size[1], self.target_img_size[0],0,0
        # возвращаем bounding_box и вероятнсть.
        for i, b in enumerate(bbx):
            if b[0] > 0: # необходимо заменить на вероятность.
                b *= self.win_size
                b[0] += self.win_coords[i][1]  
                b[1] += self.win_coords[i][0]
                b[2] += b[0]  
                b[3] += b[1]
                x = b[0] if b[0] < x else x
                y = b[1] if b[1] < y else y
                x1 = b[2] if b[2] > x1 else x1
                y1 = b[3] if b[3] > y1 else y1
        return [x,y,x1-x,y1-y]

    # bb = combined_bb(win_coords, predict)
    # rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')

    # fig,ax = plt.subplots(1)
    # ax.imshow(img, 'gray', vmin=0, vmax=1,)
    # ax.add_patch(rect)
def show_result(sample_imgs,sample_bbx):
    fig = plt.figure(figsize=(15, 15))
    for j in range(len(sample_imgs)):
        ax = fig.add_subplot(steps_y, steps_x, j+1)
        ax.imshow(sample_imgs[j])
        bb = sample_bbx[j]
        rect = plt.Rectangle((bb[0]*win_size, bb[1]*win_size), bb[2]*win_size, bb[3]*win_size,fill=False, color='red')
        ax.add_patch(rect)

        plt.xticks([]), plt.yticks([])
    plt.show()

bb = combined_bb(win_coords, predict)
rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')

fig,ax = plt.subplots(1)
ax.imshow(img, 'gray', vmin=0, vmax=1,)
ax.add_patch(rect)