# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:10:27 2021

@author: nirdo
"""

import os
import numpy as np

#import keras
import tempfile

from keras.models import model_from_json
from keras_preprocessing.image import img_to_array, load_img
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
#model = load_model('model_a.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
   
    image = load_img(img_path, target_size=(200, 200))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    #image = preprocess_input(image)
    preds = model.predict(image)
   
    return preds



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        
        # get the file from the HTTP-POST request
        f = request.files['file']        
       
        # save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #print(basepath)
        #file_path = os.path.join(basepath, 'uploads', f.filename)
        #print(file_path)
       
        filename = secure_filename(f.filename)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        
        f.save(file_path)
        
        # make prediction about this image's class
        preds = model_predict(file_path)
        
        #pred_class = decode_predictions(preds, top=10)
        #result = str(pred_class[0][0][1])
        #print('[PREDICTED CLASSES]: {}'.format(pred_class))
        print('[RESULT]: {}'.format(preds))
        
        if preds[0][0] == 1:
            prediction = 'covid'
    
        elif preds[0][1]==1:
            prediction = 'normal'
        else:
            prediction='pnumonia'
        os.remove(file_path)
        
        return render_template('index.html',prediction_text=prediction)
        #return prediction
    
    return None

if __name__ == '__main__':
    app.run(debug=False)