from flask import Flask, request, render_template
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import re
import base64
import os

with tf.device('/cpu:0'):

    app = Flask(__name__)
    model = load_model(filepath=os.path.join('models' + '/h5', 'vgg_16_model.h5'), custom_objects={'LeakyReLU': LeakyReLU})
    model.make_predict_function()

    @app.route("/")
    def home():
        return render_template("index.html")


    def convertImage(imgData1):
        imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
        with open('output.png', 'wb') as output:
            output.write(base64.b64decode(imgstr))

    @app.route("/predict", methods=['GET', 'POST'])
    def predict():

        if request.method == "POST":

            data = request.files["file"]
            data.save("img.jpg")
            
            img = load_img('img.jpg',target_size=(128, 128))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)
            classes = model.predict(img_data)

            A = np.squeeze(np.asarray(classes))
            if(A==1):
                return render_template('index.html',predicted_label="Xray contains Pneumonia")
            else:
                return render_template('index.html',predicted_label="Xray doesn't contains Pneumonia")

        return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)