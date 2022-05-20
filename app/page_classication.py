from logger import Logger
from PIL import Image
import tensorflow as tf
import os
import json
import numpy as np
import pickle


class PageClassification():

    def __init__(self, config_path, logger):
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.logger = logger
        try:

            with open(config_path, 'r', encoding='utf8') as f:
                config = json.load(f)
                self.class_names = config['page_classify_class_names']
                model_path = config['model_patch_classification']
                self.target_image_size = config['page_classify_target_image_size']
                print('Load models')

            # Загрузка модели.
            model_dict_restore = {}
            with open(model_path, 'rb') as f:
                model_dict_restore = pickle.load(f)

            self.model = tf.keras.models.model_from_json(model_dict_restore['graph'])
            self.model.set_weights(model_dict_restore['weights'])

            print('ObjectLocalization initialize')
            self.logger.write('ObjectLocalization initialize')
        except Exception as ex:
            self.logger.write(f'PageClassify_Exception: {str(ex)}')
            print(f'PageClassify_Exception: {str(ex)}')

    def get_page_type(self, image):
        '''Отправляет изображение в нейронную сеть. возвращает результат предсказания.'''
        try:
            page_info = {}
            # перед изменением размера необходимо запомнить изначальный размеры изображения.
            orign_width, orign_height = image.size

            with tf.device('CPU'):
                image = tf.keras.preprocessing.image.img_to_array(image.convert('RGB'))
                image = tf.keras.preprocessing.image.smart_resize(image, (224, 224), interpolation='bilinear')
                image = np.array([image], dtype='float') / 255.0
                pred = self.model.predict(image)

            pred_y = int(np.round(pred[0]))
            page_info['type'] = self.class_names[pred_y]
            page_info['source'] = {'width': orign_width, 'height': orign_height}
            self.logger.write(f'Page_info: {page_info}')
            print(f'Page_info: {page_info}')

        except Exception as e:
            self.logger.write(
                f'PageClassify_Exception: Ошибка интерпретации модели! Не удалось выполнить предскание. {str(e)}')
            print(f'PageClassify_Exception: Ошибка интерпретации модели! Не удалось выполнить предскание. {str(e)}')

        return page_info


if __name__ == "__main__":
    print("Walcom to ObjectLocalization!")
    file_dir = os.path.dirname(os.path.realpath('__file__'))

    loger_patch = os.path.join(file_dir, 'app/log')
    config_patch = os.path.join(file_dir, 'app/config.json')
    log = Logger(loger_patch)
    with tf.device('CPU'):
        page_classify = PageClassification(config_patch, log)
        img = Image.open(os.path.join(file_dir, '../images/obh31f00_3.tif'))
        data = page_classify.get_page_type(img)
    print(f'predicted: {data}')
