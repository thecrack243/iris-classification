# iris_classification.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score


# =====================
# 1. LOAD DATA
# =====================
def load_data(path):
    df = pd.read_csv(path)

    # Drop unnecessary column
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    return df


# =====================
# 2. PREPROCESSING
# =====================
def preprocess_data(df):
    X = df.drop(columns=["Species"])
    y = df["Species"]

    # Encoding
    y_encoded = y.map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    })

    return X, y_encoded


# =====================
# 3. SPLIT DATA
# =====================
def split_data(X, y, test_size=0.3, random_state=70):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =====================
# 4. TRAIN MODEL
# =====================
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=70)
    model.fit(X_train, y_train)
    return model


# =====================
# 5. EVALUATION
# =====================
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("")
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}")
    print(f"F1-score : {f1*100:.2f}")
    print("")

    return acc, precision, f1


# =====================
# 6. SAVE MODEL
# =====================
def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")


# =====================
# 7. MAIN PIPELINE
# =====================
def main():
    print("Loading dataset...")
    df = load_data("../data/iris_dataset.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    print("Saving model...")
    save_model(model, "../app/iris_model.pkl")

    print("\nPipeline completed successfully.")


# =====================
# RUN SCRIPT
# =====================
if __name__ == "__main__":
    main()