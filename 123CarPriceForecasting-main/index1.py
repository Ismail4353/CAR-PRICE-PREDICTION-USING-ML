from math import floor
from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import locale

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel3.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route("/")
def home():
    return render_template("index_copy.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('predict.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predictres', methods=['POST'])
@cross_origin()
def predictres():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Convert input values to the correct data types
    year = int(year)
    driven = int(driven)

    # Ensure that the input values are valid
    if company == 'Select Company' or car_model == '' or year < 1900 or year > 2100 or driven < 0 or fuel_type not in ['Petrol', 'Diesel', 'CNG']:
        return 'Invalid input values'

    # Make a prediction using the loaded model
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))

    # Format the prediction as a string with two decimal places and Indian Rupee currency symbol
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
    prediction_str = locale.currency(prediction[0], grouping=True)

    return prediction_str

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
