import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load model using correct relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")
model = joblib.load(MODEL_PATH)

# Prediction function
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    
    prediction = model.predict(features)[0]

    mapping = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    }

    return mapping[prediction]

# Function to compute prediction probability (confidence)
def predict_confidence(sepal_length, sepal_width, petal_length, petal_width):
    features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    
    probs = model.predict_proba(features)[0]
    confidence = max(probs)

    return confidence * 100

# Function to compute confusion matrix and classification report
def confusion_matrix_and_report(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    
    return cm, cr