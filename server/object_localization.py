import tensorflow as tf
from logger import Logger
# import numpy as np
import os
import json
import image_operations
from PIL import Image
import IoU

class ObjectLocalization():

    def __init__(self, config_patch, logger): # Написать стоит и други процедуры после инициализации.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.logger = logger
        try:
            
            with open(config_patch, 'r', encoding='utf8') as f:
                config = json.load(f)
                self.class_names = config['class_names']
                model_path_logo = config['model_patch_logo']
                model_path_sign = config['model_patch_sign']
                self.target_image_size = config['target_image_size']
                self.min_object_size = config['min_object_size']
                self.threshold = config['threshold']  
            print('Load models')
            # Загрузим модель.
            self.model_logo = tf.keras.models.load_model(model_path_logo, custom_objects={"IoU":IoU})
            self.model_sign = tf.keras.models.load_model(model_path_sign, custom_objects={"IoU":IoU})
           
            self.image_operations = image_operations.ImageOperations(logger, config_patch)
            self.orign_width = 0
            self.orign_height = 0

            print('ObjectLocalization initialize')
            self.logger.write('ObjectLocalization initialize')
        except Exception as ex:
            self.log.write(f'ObjectLocalization_Exception: {str(ex)}')


    def get_boxes_location(self, pred):
        '''Определим местоположение ограничивающего прямоугольника на изображении.'''
        class_name = ''
        max_value = pred.max()
        print(max_value)
        if max_value < self.threshold:
            class_name='void' # изображение не содержит ни логотипов ни подписей.
        else:
            # Загрузим изображение.
            pred_y = tf.argmax(pred, axis=1, output_type=tf.int32)
            class_name = self.class_names[pred_y[0]]
        return class_name
    
    def predict_interpretation(self, predict, class_name):
        '''Выполняет интерпредацию предсказанного значения.'''
        position={}
        print(predict[0][:,0].max())
        if predict[0][:,0].max() > self.threshold:

            bbx = self.image_operations.combined_bb(predict[0], predict[1], class_name, self.threshold)   # Объединяем боксы с разных тайлов в один.
            if (bbx[2] > self.min_object_size) and (bbx[3]> self.min_object_size): 
                bbx = self.image_operations.get_original_size_bb(bbx,(self.orign_width, self.orign_height))
              # Здесь, по идее, можно было бы создать отдельный класс и передать в него свойства. А, затем как-то его сериализовать.
                position['type']= class_name  
                position['position'] = {'left':bbx[0],'top':bbx[1],'width':bbx[2],'height':bbx[3]} # координаты объекта
                position['source'] = {'width':self.orign_width, 'height': self.orign_height}
        return position 


    def get_objects_localization(self, image):
        '''Отправляет изображение в нейронную сеть. возвращает результат предсказания.'''
        object_list = []
        try:
            # перед изменением размера необходимо запомнить изначальный размеры изображения.
            self.orign_width, self.orign_height = image.size 
            image_list = self.image_operations.image_prepare(image, self.target_image_size, split_image=True)

            pred = self.model_logo.predict(image_list)      
            position = self.predict_interpretation(pred, self.class_names[0])
            if position != {}:
                object_list.append(position)
                self.logger.write(f'logo position-{position}')
                print(f'logo position-{position}')

            pred = self.model_sign.predict(image_list)     
            position = self.predict_interpretation(pred, self.class_names[1])
            if position != {}:
                object_list.append(position)
                self.logger.write(f'signature position-{position}')
                print(f'signature position-{position}')

        except Exception as e:
            self.logger.write(f'Exception: Ошибка интерпретации модели! Не удалось выполнить предскание. {str(e)}')
            print(f'ObjectLocalization_Exception: Ошибка интерпретации модели! Не удалось выполнить предскание. {str(e)}')

        if object_list=={}:    # если словарь все еще пуст - возвращаем 'void'  
                object_list.append({'type': 'void'})
                print('Изображение не содержит ни логотипов ни подписей')
        return object_list
    

if __name__ == "__main__":

    print(("Walcom to ObjectLocalization!"))
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    # classes_path = os.path.join(file_dir,'models/class_names.txt')
    model_path_logo = os.path.join(file_dir,'models/model_logo_resnet_100.h5')
    model_path_sign = os.path.join(file_dir,"models/model_sign_resnet.h5",)
    loger_patch = os.path.join(file_dir,'server/log')
    img = Image.open(os.path.join(file_dir,'images/nuz52d00.tif'))
    class_names = ['logo','sign']
    win_size=224
    threshold=0.8
    target_image_size= (600, 800)
    log = Logger(loger_patch)
    object_localization = ObjectLocalization(model_path_logo, model_path_sign, class_names, log, win_size, target_image_size, threshold) # Для его активации нужно настроить модель.
    data = object_localization.get_objects_localization(img) 
    print(f'predicted: {data}')
    # Загружаем изображение. Пытаемся определить класс к которому оно принадлежит.