# Тестовое задание на должность Backend-разработчик сервисов.

**ML:** opencv, pillow, keras, pandas, numpy.

**API:** flask

## Задание

Задание заключается в реализации небольшого, желательно RESTful сервиса по обработке изображений и текста. Без UI. Через HTTP. Все ответы – в формате JSON. Под капотом – минимум Python 3.7, что будет в качестве веб-сервера – на ваше усмотрение.

### 1. Поиск объектов

На изображении страницы необходимо найти местоположение логотипов и подписей. Для их поиска можно использовать как классические (алгоритмические) подходы computervision, так и подходы с использованием нейронных сетей.

На вход подается файл-изображение в формате **\*.tif**. Адрес API и названия параметров для входных данных придумайте самостоятельно.

#### Данные для обучения

Вместе с заданием прилагается набор PNG-файлов с логотипами и подписями, а также XML-файлы, содержащие информацию о местоположении объектов. Эти данные можно использовать для обучения моделей и проверки реализации. XML-файлы содержат информацию в следующем виде:

``` xml
<?xml version="1.0" encoding="UTF-8"?>

<DL_DOCUMENT src="aah97e00-page02_1.tif" NrOfPages="1" docTag="xml">

<DL_PAGE gedi_type="DL_PAGE" src="aah97e00-page02_1.tif" pageID="1" width="2560" height="3296">

<DL_ZONE gedi_type="DLLogo" id="None" col="1074" row="18" width="374" height="219"> </DL_ZONE>

</DL_PAGE>

</DL_DOCUMENT>

</GEDI>
```

Информация о расположении объектов с типом **gedi_type** находится в поле **DL_ZONE****, где **col** и **row** – верхний левый угол обрамляющего прямоугольника, **width** и **height** – ширина и высота обрамляющего прямоугольника.

Изображения для обучения расположены в папке **pages/source**, а XML-файлы с информацией о логотипах в **pages/truth**. Для самопроверки рекомендуется разделить эти данные на два набора – для обучения и для тестирования.

#### Выходные данные

В ответе должна содержаться информацию о местоположении объектов. Местоположение задается с помощью верхнего левого угла, ширины и высоты обрамляющего прямоугольника. Пример:

``` json
{
 "type": "logo",
 "position": 
  {
      "left": 5,
      "top": 10,
      "width": 20,
      "height": 30
     },
 "source": 
  {
      "width": 250,
      "height": 250
  }
}
```

Для логотипов используйте тип **logo**, для подписей – **sign**. Не забудьте вернуть размеры оригинала. Все координаты и размеры указываются в пикселях.


## Описание результатов. 

Изучив датасет, обнаружил такую неприятную особенность в разметке данных.
У нас имеется два типа документов. Те что содержат логотипы и, те что содержат подписи. Одно, как бы исключает другое. Т.е. эти условия взаимоисключающие. Но, по факту все совсем не так. Изображение может содержать и то и другое. Просто один из классов не размечен. Для того чтобы избежать проблем с обучением модели, решил просто создать две модели каждая из которых будет определять координаты объектов своего класса.
Jpyter-ноутбуки с полным описанием всего процесса обучения находятся в папке **experiments**. Файлы обученными моделями находятся в **models**.модельnotebook со всеми опреациями по обучени модели. 

Список необходимых библиотек расположен в файле `requirements.txt`. 
Для установки выполнить

``` bash
pip install -r requirements.txt
```

Вопреки требованиям задания сохранил модель в формате *.h5. Т.к. сохранении в pickle возникает ошибка. Ибо модель слишком велика для данного формата. 

>:pushpin: Обученные модели находятся можно скачать архивом по ссылке. :point_right:  [Ссылка](https://drive.google.com/file/d/14pSJHegd1rF2lgZ-Mw44a3CpogRJnm6o/view?usp=sharing) :rocket:
Распакованные файлы необходимо поместить в папку /Directum_test_task/models. 

#### 6. Проверяем, что наше api работает
1. Запустить сервер. Файл - run_server.

2. Выполнить код из ноутбука - Test_doc_analizer.ipynb. :smile:



