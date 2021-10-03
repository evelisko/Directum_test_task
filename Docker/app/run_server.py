# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

import json
import os
from flask import request, jsonify, Flask, redirect
from logger import Logger
from object_localization import ObjectLocalization
from page_classication import PageClassification
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = ''
app = Flask(__name__)

dataframe_path = ""
loger_patch = ""
config = {}
config_patch = "/app/app/config.json"

with open(config_patch, 'r', encoding='utf8') as f:
    config = json.load(f)
    loger_patch = config['loger_patch']

log = Logger(loger_patch)
log.write('Run Flask Server')
print('Run Flask Server')

object_localization = ObjectLocalization(config_patch, log)  # Для его активации нужно настроить модель.
page_classify = PageClassification(config_patch, log)  # Для его активации нужно настроить модель.


@app.route("/", methods=["GET"])
def general():
    log.write("Welcome to food prediction class process. Please use 'http://<address>/service' to POST")
    return """Welcome to food prediction class process. Please use 'http://<address>/service' to POST"""


# Здесь выполним обработку POST/GET запроса.
@app.route("/service", methods=["POST", "GET"])
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

            img = request.files['file']  # Его мы и будем сохранять.

            request_json = json.loads(request.files['json'].read())

            if request_json["method"]:
                method_name = request_json["method"]
                print("method_name: ", method_name)

                if method_name == "object_localization":
                    data["object_localization"] = object_localization.get_objects_localization(Image.open(img))

                if method_name == "page_cassify":
                    data["page_cassify"] = page_classify.get_page_type(Image.open(img))

                print(f'data: {data}')
        except Exception as e:
            log.write(f'Exception: {str(e)}')
            print(f'Exception: {str(e)}')
            data["success"] = False
    return jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
