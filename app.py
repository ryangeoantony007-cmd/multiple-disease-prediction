
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['POST'])
def diabetes():
    data = [float(x) for x in request.form.values()]
    prediction = diabetes_model.predict([data])[0]
    return render_template('result.html', disease='Diabetes', result=prediction)

@app.route('/heart', methods=['POST'])
def heart():
    data = [float(x) for x in request.form.values()]
    prediction = heart_model.predict([data])[0]
    return render_template('result.html', disease='Heart Disease', result=prediction)

@app.route('/parkinsons', methods=['POST'])
def parkinsons():
    data = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([data])[0]
    return render_template('result.html', disease='Parkinsons', result=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

