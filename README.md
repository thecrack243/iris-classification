# Iris Flower Classification

A machine learning project that classifies iris flowers into three species using their physical measurements.

This project demonstrates a complete machine learning workflow including data exploration, model training, evaluation, and visualization, with an optional interactive Streamlit interface.



## Dataset

The Iris dataset contains 150 samples with the following features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)
- Species (target label)

### Classes
- Iris-setosa  
- Iris-versicolor  
- Iris-virginica  



## Objective

The goal of this project is to predict the species of an iris flower based on its physical measurements using machine learning.



## Model

A **Random Forest Classifier** is used for training and prediction.

### Why Random Forest:
- High accuracy on structured datasets
- Handles non-linear relationships well
- Robust and stable performance
- Minimal preprocessing required



## Features

- Data exploration and visualization (2D & 3D plots)
- Machine learning model training and evaluation
- Confusion matrix analysis
- Feature importance visualization
- Interactive prediction system (Streamlit)
- PDF report generation



## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib



## Project Structure

```
iris_classification/
│
├── app/
│   │   app.py
│   │   iris_model.pkl
│   │   utils.py
│
├── data/
│   │   iris_dataset.csv
│
├── img/
│   │   Iris_setosa.jpg
│   │   Iris_versicolor.jpg
│   │   Iris_virginica.jpg
│   │
│   └── output/
│           class_dist.png
│           conf_matrix.png
│           corr_matrix.png
│           data_viz.png
│           desc_bound.png
│           eda.png
│           pca.png
│           violin_plot.png
│
├── notebooks/
│       iris_classification.ipynb
│
├── src/
│       iris_classification.py
│
│
│   .gitignore
│   README.md
│   requirements.txt
```



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thecrack243/iris-classification.git
cd iris_classification
```

### 2. (Optional) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Train the model (optional)

```bash
cd src
python iris_classification.py
```

### Run the Streamlit app

```bash
cd app
streamlit run app.py
```




## Results

The model achieves strong performance on the dataset:

- High accuracy on test data
- Excellent class separation
- Petal features are the most important predictors



## Key Insights

- Petal length and petal width are the most important features
- The dataset is highly separable
- Random Forest performs very well without heavy tuning



## Future Improvements

- Add more models (SVM, KNN comparison)
- Hyperparameter tuning
- Improve UI/UX design



## Author

Emmanuel Ilunga  
Machine Learning Project — 2026  



## License

This project is open-source and available for educational and learning purposes.