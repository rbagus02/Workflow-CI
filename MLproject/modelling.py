import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Load data
X_test = pd.read_csv('iris_preprocessing/X_test.csv')
X_train = pd.read_csv('iris_preprocessing/X_train.csv')
y_test = pd.read_csv('iris_preprocessing/y_test.csv')
y_train = pd.read_csv('iris_preprocessing/y_train.csv')

# 2. Setup MLflow
mlflow.set_experiment("Iris Klasifikasi Random Forest")
mlflow.sklearn.autolog()

# 3. Train model dengan Random Forest
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)