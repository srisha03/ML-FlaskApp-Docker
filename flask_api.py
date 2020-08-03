# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020

@author: srisha
"""

from flask import Flask, request
import pickle
import pandas as pd
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("rf_classifier.pkl","rb")
rf_classifier = pickle.load(pickle_in)

# by default we have get methods
@app.route('/')
def welcome():
    return "Welcome to the Flask App"


@app.route('/predict')
def predict_note_authentication():
   
    """Authenticator for Bank Notes
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    predicted_class = rf_classifier.predict([[variance,skewness, curtosis, entropy]])
    return "The predicted class is: " + str(predicted_class)


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Authenticator for Bank Notes
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = rf_classifier.predict(df_test)
    return "The prediscted class fot the rocords are: " + str(list(prediction))
    
# @app.route('/predict_file',methods=["POST"])
# def predict_note_file():
#     #print(request)
#     #print(request.files)
#     df_test = pd.read_csv(request.files.get("file"))
#     prediction = rf_classifier.predict(df_test)
#     return "The predicted class for the records are: " + str(list(prediction))

if __name__=='__main__':
    # app.debug = True
    # app.run()
    app.run(host='0.0.0.0',port=5000)
    
    
    