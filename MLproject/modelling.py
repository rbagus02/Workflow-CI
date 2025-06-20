import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
import os

def main():
    # 1. Load data
    df = pd.read_csv('penguins_preprocessing.csv')

    # 2. Prepare train-test split
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    # 3. MLflow setup - hanya set experiment jika tidak dijalankan via CLI
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("penguins-klasifikasi")

    # 4. Gunakan nested_run jika run sudah ada dari CLI
    if os.environ.get("MLFLOW_RUN_ID"):
        with mlflow.start_run(nested=True):
            train_and_log_model(X_train, X_test, y_train, y_test)
    else:
        with mlflow.start_run():
            train_and_log_model(X_train, X_test, y_train, y_test)

def train_and_log_model(X_train, X_test, y_train, y_test):
    # Model training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for label in report:
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            mlflow.log_metric(f"precision_{label}", report[label]['precision'])
            mlflow.log_metric(f"recall_{label}", report[label]['recall'])
            mlflow.log_metric(f"f1_{label}", report[label]['f1-score'])

if __name__ == "__main__":
    main()