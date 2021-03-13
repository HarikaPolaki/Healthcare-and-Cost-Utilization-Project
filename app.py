#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

#Initialize the flask App
app = Flask(__name__, static_url_path = "", static_folder = "img")
model = pickle.load(open('los_diag_lin.pkl', 'rb'))
socio_model = pickle.load(open('withSocio.pkl', 'rb'))

totchg_diag = pickle.load(open('totchg_diag_lin.pkl', 'rb'))
totchg_socio = pickle.load(open('totchg_socio.pkl', 'rb'))
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    F0 = request.form['F0']
    F1 = request.form['F1']
    F2 = request.form['F2']
    F3 = request.form['F3']
    F4 = request.form['F4']
    F5 = request.form['F5']
    F6 = request.form['F6']
    F7 = request.form['F7']
    F8 = request.form['F8']
    F9 = request.form['F9']
    
    input_variables = pd.DataFrame([[F0, F1, F2, F3, F4, F5, F6, F7, F8, F9]],
                                       columns=['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                                       dtype=float)
    prediction = round(model.predict(input_variables)[0])
    diag_totchg_log = totchg_diag.predict(input_variables)[0]
    diag_totchg_pred = round(np.expm1(diag_totchg_log))
    return render_template('index.html', prediction_text='Predicted Length of stay is :{}'.format(prediction), totchg_text='Predicted Total charges is :{}'.format(diag_totchg_pred),title_info='Predictions using mental illness diagnosis only')

@app.route('/socio',methods=['POST'])
def socio():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]

    F0 = request.form['F0']
    F1 = request.form['F1']
    F2 = request.form['F2']
    F3 = request.form['F3']
    F4 = request.form['F4']
    F5 = request.form['F5']
    F6 = request.form['F6']
    F7 = request.form['F7']
    F8 = request.form['F8']
    F9 = request.form['F9']
    AGE = request.form['AGE']
    FEMALE = request.form['FEMALE']

    RACE = request.form['RACE']
    ZIPINC_QRTL = request.form['ZIPINC_QRTL']
    PAY1 = request.form['PAY1']
    HOSP_REGION = request.form['HOSP_REGION']
    HOSP_LOCTEACH = request.form['HOSP_LOCTEACH']
    
    input_variables = pd.DataFrame([[F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, AGE, FEMALE, RACE, ZIPINC_QRTL, PAY1, HOSP_REGION, HOSP_LOCTEACH]],
                                       columns=['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'AGE', 'FEMALE', 'RACE', 'ZIPINC_QRTL', 'PAY1', 'HOSP_REGION', 'HOSP_LOCTEACH'],
                                       dtype=float)
    socio_prediction = round(socio_model.predict(input_variables)[0])
    socio_totchg_log = totchg_socio.predict(input_variables)[0]
    socio_totchg_pred = round(np.expm1(socio_totchg_log))
    
  
    return render_template('index.html', prediction_text='Predicted LOS is :{}'.format(socio_prediction), totchg_text='Predicted TOTCHG is :{}'.format(socio_totchg_pred),title_info='Predictions using mental illness diagnosis and socio demographics')



if __name__ == "__main__":
    app.run(debug=True)
