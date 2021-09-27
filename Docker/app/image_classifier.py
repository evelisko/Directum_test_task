# import dill
# import flask
import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
import tensorflow as tf
import cv2
# import matplotlib.pyplot as plt
# import imageio
from logger import Logger
import numpy as np
import os
import image_prepare

class ImageClassifier(): # Здесь у нас отрабатывает модель!!! # Все основные методы располагаться будут здесь :)

    def __init__(self, model_path, classes_path, logger, win_size, target_img_size, thash_hold): # Написать стоит и други процедуры после инициализации.
        os.environ['CUDA_VISIBLE_DEVICES'] = '' 

        # self.win_size = win_size
        self.target_img_size = target_img_size
        self.trash_hold = thash_hold  

        # Загрузим модель.
        self.model = tf.keras.models.load_model(model_path)
        self.logger = logger
        self.image_prepare = image_prepare.ImagePrepare(win_size, target_img_size)
        self.orign_width = 0
        self.orign_height = 0

        with open(classes_path, 'r', encoding='utf-8') as f_obj:
            self.class_names = f_obj.read().splitlines()

    def prepare(self, image):
        image = np.array(image)
        image = cv2.cvtColor((image).astype(np.uint8), cv2.COLOR_BGR2RGB)

        # перед изменением размера необходимо запомнить изначальный размеры изображения.
        self.orign_width, self.orign_height = image.shape[0], image.shape[1] 
        image = cv2.resize(image, self.target_img_size)
        image = np.array(image, dtype=np.float32)/255.0
        image_list = self.image_prepare.split_image(image)
        return image_list     
        
    def get_bbclass_name(self, pred):
        class_name = ''
        max_value = pred.max()
        print(max_value)
        if max_value < self.trash_hold:
            class_name='void' # изображение не содержит ни логотипов ни подписей.
        else:
            # Загрузим изображение.
            pred_y = tf.argmax(pred, axis=1, output_type=tf.int32)
            class_name = self.class_names[pred_y[0]]
        return class_name

    def get_bbx(self, image):
        print('hello')
        image_list = self.prepare(image)
        pred = self.model.predict(image_list)
        bbx = self.image_prepare.combined_bb(pred)
        food_class_name = self.get_class_name(pred)
        return food_class_name
    

if __name__ == "__main__":

    print(("Walcom to image classifier!"))

    classes_path = '/home/sergey/Документы/Directum_test_task/experiments/class_names.txt'
    model_path = '/home/sergey/Документы/Directum_test_task/experiments/model.h5'
    loger_patch = '/home/sergey/Документы/Directum_test_task/experiments/log'
    img = cv2.imread('/home/sergey/Документы/Directum_test_task/images/obh31f00_3.tif')
    input_size=224
    thash_hold=0.5
    target_img_size= (600, 800)
    log = Logger(loger_patch)
    classifier = ImageClassifier(model_path, classes_path, log, input_size,target_img_size, thash_hold) # Для его активации нужно настроить модель.

    print(f'predicted: {classifier.get_class(img)}')
    # Загружаем изображение. Пытаемся определить класс к которому оно принадлежит.