import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Charger le modèle
model = joblib.load("iris_model.pkl")

# Fonction de prédiction
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

# Fonction pour la prédiction et le calcul de la confiance
def predict_confidence(sepal_length, sepal_width, petal_length, petal_width):
    features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    
    probs = model.predict_proba(features)[0]
    confidence = max(probs)

    return confidence * 100

# Fonction pour calculer la matrice de confusion et le rapport de classification
def confusion_matrix_and_report(model, X_test, y_test):
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Matrice de confusion et rapport de classification
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    
    return cm, cr
