import json
import numpy as np
import math
import tensorflow as tf


class ImageOperations():

    def __init__(self, logger, config_patch):

        self.logger = logger
        try:
            with open(config_patch, 'r', encoding='utf8') as f:
                config = json.load(f)
                self.class_names = config['obj_location_class_names']
                self.win_size = config['win_size']
                self.min_object_size = config['min_object_size']
                self.step_x = config['step_x']
                self.step_y = config['step_y']
                self.win_size = config['win_size']
                self.target_img_size = config['obj_location_target_image_size']
        except Exception as ex:
            self.logger.write(f'Image_operation Exception: {str(ex)}')
            print(f'Image_operation Exception: {str(ex)}')

        # Для того чтобы загнать катринку в нейронную сеть, необходимо разбить изображение на квадраты размером win_size.
        # следующие два параметра оптределяют к-во этих квадратов. 
        self.steps_x = math.ceil(self.target_img_size[0] / self.step_x)
        self.steps_y = math.ceil(self.target_img_size[1] / self.step_y)

        # запомнить позицию каждого квадрата.
        self.windows_coords = self.calc_window_coords()

    # Необходимо принять изображение. Определить его исходный размер.
    # уменьшить его размер до 600х800.
    # Разрезать на составные части
    # Получить ответ.
    # Объединить прямоугольники, определить их реальные координаты.
    # Пересичитатьих позицию на исходном изображении.
    # Вывести ответ в виде словаря.

    def create_mask(self, bb, x, class_index):
        """Создаем маску для bounding box'a такого же шейпа как и изображение"""
        rows, cols, _ = x.shape
        y_ind = np.zeros((rows, cols))
        bb = bb.astype(np.int)

        if class_index > 0:
            y_ind[bb[1]:bb[3] + bb[1], bb[0]: bb[2] + bb[0]] = class_index
        return y_ind

    def create_bb_array(self, x, class_index):
        """Генерируем массив bounding box'a из столбца train_df"""
        if class_index == 0:
            res = np.array([-1, -1, -1, -1])
        else:
            res = np.array([x[0], x[1], x[2], x[3]])
        return res

    def mask_to_bb(self, mask):
        """Конвертируем маску Y в bounding box'a, принимая 0 как фоновый ненулевой объект """
        # bbx = []
        # for i in range(len(class_names)-1):
        cols, rows = np.nonzero(mask)
        if len(cols) == 0:
            bb = [-1, -1, -1, -1]
        else:
            top_row = np.min(rows)
            left_col = np.min(cols)
            bottom_row = np.max(rows)
            right_col = np.max(cols)
            bb = [top_row, left_col, bottom_row - top_row, right_col - left_col]
        return np.array(bb, dtype=np.float32)

    def calc_window_coords(self):
        '''Вычисляет координаты квадратных областей, (кусков) на которые будет 
        разреанно исходное иображение.'''

        window_coords = []
        for i in range(self.steps_y):
            for j in range(self.steps_x):
                pos_x = self.step_x * j
                pos_y = self.step_y * i
                if pos_x + self.win_size > self.target_img_size[0]:
                    pos_x = self.target_img_size[0] - self.win_size
                if pos_y + self.win_size > self.target_img_size[1]:
                    pos_y = self.target_img_size[1] - self.win_size
                window_coords.append([pos_y, pos_x, self.win_size, self.win_size])
        return window_coords

    def split_image(self, img, bbx=None, class_index=1):
        '''Процедура для разбиения изображения на квадраты'''

        is_not_bbx = False
        if not bbx:
            is_not_bbx = True
            bbx = np.array([-1, -1, -1, -1], dtype='float')
        mask = self.create_mask(bbx, img, class_index)
        # разделим изображение на составные части и возращем кусочки small_imgs and small_bbx
        img_combined = np.concatenate([img, mask[..., None]], axis=2)
        X_batch = []
        y_batch = []
        for coord in self.windows_coords:
            combined = img_combined[coord[0]:(coord[0] + coord[2]), coord[1]:(coord[1] + coord[3])]
            X_batch.append(combined[..., :3])
            bb = self.mask_to_bb(combined[..., 3])
            y_batch.append(bb)

        X_batch = np.array(X_batch, dtype='float') / 255.0
        y_batch = np.array(y_batch, dtype='float') / 255.0

        if is_not_bbx:
            return X_batch
        else:
            return X_batch, y_batch

    def image_prepare(self, image, target_img_size, split_image=False):
        '''Подготавливает изображение к отправке его в нейронную сеть.'''
        try:
            with tf.device('CPU'):
                image = image.resize(target_img_size)
                image = tf.keras.preprocessing.image.img_to_array(image.convert('RGB'))

            image_list = []
            if split_image:
                print("split image")
                image_list = self.split_image(image)  # Разбиваем изображение на тайлы.

        except Exception as ex:
            self.logger.write(
                f'Image_operation Exception: Возникла ошибка на этапе предобработки изображения. {str(ex)}')
            print(f'Image_operation Exception: Возникла ошибка на этапе предобработки изображения. {str(ex)}')

        return image_list if split_image else image

    def combined_bb(self, scores, bbx, class_name, threshold=0.9):
        '''Перед отправкой в нейронную сеть изображение было разбито на более мелкие. 
           Данный метод объединяет box-ы обнаруженные на разных частях изображения в один.'''

        bounding_box = [-1, -1, -1, -1]
        if np.max(scores) < threshold:
            print('max_score: ', np.max(scores))
            return bounding_box
        top_row, left_col, bottom_row, right_col = -1, -1, -1, -1

        counter = 0
        for i, b in enumerate(bbx):
            if (scores[i] >= threshold):

                box = b * self.win_size
                box[0] += self.windows_coords[i][1] - box[2] / 2
                box[1] += self.windows_coords[i][0] - box[3] / 2
                box[2] += box[0]
                box[3] += box[1]
                print('score: ', scores[i], 'box: ', box)

                # Огромный костыль. т.к. модель не очень хорошо умеет локацию подписей. Поэтому, нужно немного ей помочь.
                if (class_name == 'sign') and (box[1] < self.target_img_size[1] / 2):
                    continue
                if (class_name == 'logo') and (box[1] > self.target_img_size[1] / 2):
                    continue
                if (np.abs(box[2] - box[0]) < self.min_object_size) or (np.abs(box[3] - box[1]) < self.min_object_size):
                    continue

                counter += 1
                if counter == 1:
                    top_row = box[0]
                    left_col = box[1]
                    bottom_row = box[2]
                    right_col = box[3]
                else:
                    if (np.abs(top_row - box[0]) > self.win_size / 2) or (
                            np.abs(left_col - box[1]) > self.win_size / 2):
                        print(np.abs(top_row - box[0]), np.abs(left_col - box[1]))
                        continue
                    top_row = box[0] if box[0] < top_row else top_row
                    left_col = box[1] if box[1] < left_col else left_col
                    bottom_row = box[2] if box[2] > bottom_row else bottom_row
                    right_col = box[3] if box[3] > right_col else right_col

        bounding_box = [top_row, left_col, bottom_row - top_row, right_col - left_col]
        return bounding_box

    def get_original_size_bb(self, bb, old_size):  # исходный размер изображения тоже нужно как-то анализировать.
        '''Приводит координаты box-ов к масштабу исходного изображения.'''
        scale_width = old_size[0] / self.target_img_size[0]
        scale_height = old_size[1] / self.target_img_size[1]
        original_bb = [0, 0, 0, 0]
        original_bb[0] = int(np.round(bb[0] * scale_width))
        original_bb[1] = int(np.round(bb[1] * scale_height))
        original_bb[2] = int(np.round(bb[2] * scale_width))
        original_bb[3] = int(np.round(bb[3] * scale_height))
        return original_bb
