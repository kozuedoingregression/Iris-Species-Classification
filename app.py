import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load data and model
try:
    data = pd.read_csv('database/Iris.xls')
    model = joblib.load("model/IrisFlowerClassification.pkl")
except Exception as e:
    print(f"Error loading data or model: {str(e)}")

def validate_input(input_data):
    """Validate input data"""
    try:
        # Convert to float and check if values are positive
        values = [float(input_data[col]) for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        return all(v > 0 for v in values)
    except (ValueError, TypeError):
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        input_values = {
                'SepalLengthCm': request.form.get('SepalLengthCm'),
                'SepalWidthCm': request.form.get('SepalWidthCm'),
                'PetalLengthCm': request.form.get('PetalLengthCm'),
                'PetalWidthCm': request.form.get('PetalWidthCm')
                }

        # Validate input
        if not all(input_values.values()):
            return "Missing input values", 400

        if not validate_input(input_values):
            return "Invalid input values", 400

        # Create DataFrame for prediction
        input_data = pd.DataFrame([input_values])

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            return "Predicted specie: Iris-setosa"
        elif prediction == 1:
            return "Predicted specie: Iris-versicolor"
        else:
            return "Predicted specie: Iris-virginica"

    except Exception as e:
        return f"Error making prediction: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True, port=5001)
