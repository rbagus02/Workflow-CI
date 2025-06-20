import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
import os
import sys

def main():
    # 1. Handle file path with multiple fallback options
    file_path = (
        sys.argv[1] if len(sys.argv) > 1 
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "penguins_preprocessing.csv")
    )
    
    # Additional fallback paths
    possible_paths = [
        file_path,
        'penguins_preprocessing.csv',
        'MLproject/penguins_preprocessing.csv',
        '../MLproject/penguins_preprocessing.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            break
        except Exception as e:
            print(f"Failed to load from {path}: {str(e)}")
            continue

    if df is None:
        raise FileNotFoundError(f"Could not find CSV file in any of these locations: {possible_paths}")

    # 2. Prepare data
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    # 3. MLflow setup
    mlflow.set_experiment("penguins-klasifikasi")
    
    with mlflow.start_run():
        # 4. Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            verbose=1
        )
        model.fit(X_train, y_train)
        
        # 5. Evaluate and log
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "test_size": 0.2,
            "random_state": 42,
            "data_source": path  # Log which file was actually used
        })
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1-score": report['weighted avg']['f1-score']
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()