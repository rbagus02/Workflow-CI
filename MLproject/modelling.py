import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
import os

# 1. Load data dari path yang sesuai dengan environment variable
csv_path = os.path.join('MLproject', 'penguins_preprocessing.csv')  # Sesuaikan dengan struktur folder
df = pd.read_csv(csv_path)

# 2. Siapkan train-test split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

# 3. Nonaktifkan autolog dan kelola run secara manual
mlflow.set_experiment("penguins-klasifikasi")

with mlflow.start_run():
    # 4. Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    
    # 5. Evaluasi dan log manual
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log parameter
    mlflow.log_params({
        "n_estimators": 100,
        "test_size": 0.2,
        "random_state": 42
    })
    
    # Log metrik
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score']
    })
    
    # Simpan model
    mlflow.sklearn.log_model(model, "model")