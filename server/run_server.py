# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

import json
import os
# import dill
from flask import request, jsonify, Flask, redirect
from logger import Logger
from object_localization import ObjectLocalization
# import cv2
from PIL import Image 

# import the necessary packages
# dill._dill._reverse_typemap['ClassType'] = type

# initialize our Flask application and the model
app = Flask(__name__)
    
dataframe_path = ""
loger_patch = ""
config = {}
config_patch = "server/config.json"

# file_dir = os.path.dirname(os.path.realpath('__file__'))
# classes_path = os.path.join(file_dir,'experiments/class_names.txt')

# print('current_dir',file_dir)
with open(config_patch, 'r', encoding='utf8') as f:
    config = json.load(f)
    # dataframe_path = config['dataframe_path']
    model_path_logo = config['model_path_logo']
    model_path_sign = config['model_path_sign']
    classes_path = config['class_names']
    

    win_size = config['win_size']
    target_image_size = config['target_image_size']
    threshold = config['threshold']
    
    loger_patch = config['loger_patch']

log = Logger(loger_patch)
log.write('Run Flask Server')
print('Run Flask Server')

object_localization = ObjectLocalization(model_path_logo, model_path_sign, classes_path, logger=log, win_size=win_size, target_image_size=target_image_size, threshold=threshold) # Для его активации нужно настроить модель.

@app.route("/", methods=["GET"])
def general():
    log.write("Welcome to food prediction class process. Please use 'http://<address>/service' to POST")
    return """Welcome to food prediction class process. Please use 'http://<address>/service' to POST"""

# Нужно по старинке файл отправить и посмотреть как сервер отработает!

@app.route("/service", methods=["POST","GET"])  # Здесь выполним обработку post запроса.
def netflix_films():
    data = {"success": False}
    if request.method == "POST":
        try:
            data["success"] = True
            if 'json' not in request.files:
                print('No json file part')
                return redirect(request.url)
              
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)

            img = request.files['file'] # Его мы и будем сохранять.
            
            request_json = json.loads(request.files['json'].read())

            if request_json["method"]:
                method_name = request_json["method"]
                print("method_name: ",method_name)
                if method_name == "object_localization":
                    print('zzzzz')
                    # print(img.read())
                    data["object_localization"] = object_localization.get_objects_localization(Image.open(img)) # Возвращаем результат классификации
                    # img.save(f'/home/sergey/Документы/Food_Classifier/img/{img.filename}')
                    print(f'real_title: {img.filename}')
                    # img = Image.open(file)
                print(f'data: {data}')
        except Exception as e:
            log.write(f'Exception: {str(e)}')
            print(f'Exception: {str(e)}')
            data["success"] = False
    # return the data dictionary as a JSON response
    return  jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
