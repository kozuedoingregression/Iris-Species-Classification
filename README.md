# Iris Species Classification Web App 🌸

A Flask-based web application that predicts Iris flower species using machine learning. The application uses a trained model to classify Iris flowers into three species: Setosa, Versicolor, and Virginica, based on their sepal and petal measurements.

## Features ✨

- Interactive web interface for input measurements
- Real-time predictions using machine learning
- Simple and intuitive design

## Demo 🚀

![Application Demo](/demos/demo.gif)

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/kozuedoingregression/Iris-Species-Classification.git
cd Iris-Species-Classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Start the Flask application:
```bash
python app.py
```

2. Enter the following measurements in centimeters:
   - Sepal Length
   - Sepal Width
   - Petal Length
   - Petal Width

3. Click "Predict" to see the classification result

4. Run tests
```bash
python -m unittest testing.py
```

## Project Structure 📁

```
iris-classification/
├── database/
│   ├── Iris.xls
├── model/
│   ├──IrisFlowerClaffification.pkl
├── templates/
│   ├── index.html
├── app.py              # Flask application
├── requirements.txt
├── testing.py
```
## Model Performance 📊

- The model was trained using scikit-learn's Logistic Regression on the [Iris Dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv).
- Accuracy: 97%
- [NoteBook](https://www.kaggle.com/code/shashanknecrothapa/iris-flower-classification)
  
## Technical Details 🔧

- **Framework**: Flask
- **Machine Learning**: scikit-learn
- **Model**: Logistic Regression
- **Dataset**: [Iris Dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- **Frontend**: HTML, CSS, JavaScript


## Requirements 📋

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy

## Development 👨‍💻

To contribute to this project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Contact 📧

For questions or feedback, please contact:
- X: [kozue](https://x.com/0xaa248)
- GitHub: [kozuedoingregression](https://github.com/kozuedoingregression)
