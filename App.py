from imageai.Detection import ObjectDetection, keras_retinanet
import os
from PIL import Image
import base64
import logging as log
from flask import Flask, request, make_response
from flask_restful import Resource, Api
import flask_restful as restful
from keras.engine.saving import load_model
from sqlalchemy import create_engine
from json import dumps
from flask import jsonify
from flask_classful import FlaskView
import json
from keras import backend as K
import tensorflow as tf
from tkinter.filedialog import askopenfilename
from pathlib import Path
from tkinter import Tk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
result = dict()


def init():
    global model, graph
    # load the pre-trained Keras model
    # model = load_model('resnet50_coco_best_v2.0.1.h5')
    graph = tf.get_default_graph()

Tk().withdraw()
filename = askopenfilename()
img = Path(filename).name
with open(filename, "rb") as imageFile:
    # converting download.jpg to a String
    str_file = base64.b64encode(imageFile.read())
    print(str_file)


app = Flask(__name__, template_folder='template')


@app.route("/", methods=["GET","POST"])
def predict():
    with graph.as_default():
        fh = open("imageToSave.png", "wb")
        fh.write(base64.decodestring(str_file))
        fh.close()
        log.debug("File decrypted !!")
        filename = "imageToSave.png"

        log.basicConfig(filename="logs.log", level=log.DEBUG)
        # giving the filename
        # filename = "image.jpg"
        execution_path = os.getcwd()

        # creating the detector object for ObjectDetection
        log.info("Detector activated ")
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=filename,
                                                     output_image_path=os.path.join(execution_path, "imagenew.jpg"))

        # printing the found object names and the probability value
    result = dict()

    for eachObject in detections:
        # print(eachObject["name"], " : ", eachObject["percentage_probability"])
        result.update({eachObject["name"]: eachObject["percentage_probability"]})

    print(result)
    return sendResponse(result)
    K.clear_session()
    # displaying image after object detection
    img = Image.open('imagenew.jpg')
    img.show()

    # passing an string value of image to object
    # object detect class gets inherited from imageconvert

    if request.method=='POST':
       return result

@app.route('/form', methods=['GET', 'POST'])
def form_example():
    if request.method == 'POST':
        with graph.as_default():
            fh = open("imageToSave.png", "wb")
            fh.write(base64.decodestring(str_file))
            fh.close()
            log.debug("File decrypted !!")
            filename = "imageToSave.png"

            log.basicConfig(filename="logs.log", level=log.DEBUG)
            # giving the filename
            # filename = "image.jpg"
            execution_path = os.getcwd()

        # creating the detector object for ObjectDetection
            log.info("Detector activated ")
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            detections = detector.detectObjectsFromImage(input_image=filename,
                                                         output_image_path=os.path.join(execution_path, "imagenew.jpg"))

            # printing the found object names and the probability value
        result = dict()

        for eachObject in detections:
            # print(eachObject["name"], " : ", eachObject["percentage_probability"])
            result.update({eachObject["name"]: eachObject["percentage_probability"]})

        print(result)
        return sendResponse(result)
        K.clear_session()

    return '''<form method="POST">
                  Image string: <input type="text" name="imagestr"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


def sendResponse(responseObj):
    response = jsonify(responseObj)
    return response



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    init()
    app.run(threaded=True, debug="on", port=9000)
