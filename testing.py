import unittest
import pandas as pd
import numpy as np
import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, model, data
import json

class IrisClassificationTests(unittest.TestCase):
    def setUp(self):
        """Set up test client and test data"""
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Test data for different species
        self.setosa_data = {
                'SepalLengthCm': '5.1',
                'SepalWidthCm': '3.5',
                'PetalLengthCm': '1.4',
                'PetalWidthCm': '0.2'
                }

        self.versicolor_data = {
                'SepalLengthCm': '6.4',
                'SepalWidthCm': '2.9',
                'PetalLengthCm': '4.3',
                'PetalWidthCm': '1.3'
                }

        self.virginica_data = {
                'SepalLengthCm': '7.7',
                'SepalWidthCm': '3.8',
                'PetalLengthCm': '6.7',
                'PetalWidthCm': '2.2'
                }

        self.invalid_data = {
                'SepalLengthCm': 'invalid',
                'SepalWidthCm': '3.5',
                'PetalLengthCm': '1.4',
                'PetalWidthCm': '0.2'
                }

    def test_home_page(self):
        """Test if home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_setosa(self):
        """Test prediction for Iris-setosa"""
        response = self.client.post('/predict', data=self.setosa_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted specie: Iris-setosa', response.data)

    def test_predict_versicolor(self):
        """Test prediction for Iris-versicolor"""
        response = self.client.post('/predict', data=self.versicolor_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted specie: Iris-versicolor', response.data)

    def test_predict_virginica(self):
        """Test prediction for Iris-virginica"""
        response = self.client.post('/predict', data=self.virginica_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted specie: Iris-virginica', response.data)

    def test_missing_data(self):
        """Test prediction with missing data"""
        incomplete_data = {
                'SepalLengthCm': '5.1',
                'SepalWidthCm': '3.5'
                }
        response = self.client.post('/predict', data=incomplete_data)
        self.assertEqual(response.status_code, 400)

    def test_invalid_values(self):
        """Test prediction with invalid values"""
        response = self.client.post('/predict', data=self.invalid_data)
        self.assertEqual(response.status_code, 400)

    def test_model_loaded(self):
        """Test if model is loaded correctly"""
        self.assertIsNotNone(model)

    def test_data_loaded(self):
        """Test if data is loaded correctly"""
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_extreme_values(self):
        """Test prediction with extreme values"""
        extreme_data = {
                'SepalLengthCm': '100',
                'SepalWidthCm': '100',
                'PetalLengthCm': '100',
                'PetalWidthCm': '100'
                }
        response = self.client.post('/predict', data=extreme_data)
        self.assertEqual(response.status_code, 200)

    def test_negative_values(self):
        """Test prediction with negative values"""
        negative_data = {
                'SepalLengthCm': '-5.1',
                'SepalWidthCm': '-3.5',
                'PetalLengthCm': '-1.4',
                'PetalWidthCm': '-0.2'
                }
        response = self.client.post('/predict', data=negative_data)
        self.assertEqual(response.status_code, 400)

    def test_zero_values(self):
        """Test prediction with zero values"""
        zero_data = {
                'SepalLengthCm': '0',
                'SepalWidthCm': '0',
                'PetalLengthCm': '0',
                'PetalWidthCm': '0'
                }
        response = self.client.post('/predict', data=zero_data)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
