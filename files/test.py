import pandas as pd
import numpy as np
from flask import Flask, request, send_file, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('pipe.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    file=request.files['file']
    data = pd.read_csv(file)

    col_names = data.columns.values.tolist()
    col_sorted = col_names.sort()
    list = ['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    list_sort = list.sort()
    if(col_sorted == list_sort):
        temp = data['Temperature'].mean()
        hum = data['Humidity'].mean()
        light = data['Light'].mean()
        co2 = data['CO2'].mean()
        hr = data['HumidityRatio'].mean()

        data = data.fillna(value={
            'Temperature':temp,
            'Humidity':hum,
            'Light':light,
            'CO2':co2,
            'HumidityRatio':hr
        })

        new_file=data.drop(['date', 'HumidityRatio'], axis=1)

        prediction=model.predict(new_file)
        data['Occupancy']=prediction
        final = data.to_csv('roomoccupancy.csv', index=False)

        return render_template('index.html', prediction_text='Your output file is downloaded in your device named as "roomoccupancy.csv"')
    else:
        return render_template('index.html', prediction_text='Uploaded file has insufficient column')
    


    # return render_template('index.html', prediction_text='Your output file is downloaded in your device named as "roomoccupancy.csv"')

if __name__ == "__main__":
    app.run(debug=True)

