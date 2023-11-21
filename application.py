import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

application = Flask(__name__)
data = pd.read_csv('database/Iris.xls')
model = joblib.load("model/IrisFlowerClassification.pkl")


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    SepalLengthCm = request.form.get('SepalLengthCm')
    SepalWidthCm = request.form.get('SepalWidthCm')
    PetalLengthCm = request.form.get('PetalLengthCm')
    PetalWidthCm = request.form.get('PetalWidthCm')

    input_data = pd.DataFrame([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]], columns=[
                              'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    prediction = model.predict(input_data)[0]
    print(prediction)
    if (prediction == 0):
        return "Predicted specie: Iris-setosa"
    elif (prediction == 1):
        return "Predicted specie: Iris-versicolor"
    else:
        return "Predicted specie: Iris-virginica"

if __name__ == "__main__":
    application.run(debug=True, port=5001)
