from flask import Flask, render_template, request, url_for
from PIL import Image
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('modeluts.h5')

class_labels = ['AFRICAN LEOPARD', 'CARACAL', 'CHEETAH', 'CLOUDED LEOPARD', 'JAGUAR',
                'LIONS', 'OCELOT', 'PUMA', 'SNOW LEOPARD', 'TIGER']

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((244, 244))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_animal(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence

# @app.route('/', methods=['GET','POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             prediction, confidence = predict_animal(file_path)
#         # return  render_template('result.html', Image.file-file.filename, prediction-prediction, confidence-round(confidence))
#         return render_template('result.html', file=file.filename, prediction=prediction, confidence=confidence)

#         # return render_template('result.html', file=file.filename, prediction=prediction, confidence=confidence)
#     return  render_template('upload.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction, confidence = predict_animal(file_path)
            return render_template('result.html', file=file.filename, prediction=prediction, confidence=confidence)
        else:
            return render_template('upload.html', error="Tidak ada file yang dipilih.")
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)