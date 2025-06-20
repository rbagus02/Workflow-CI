import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv('penguins_preprocessing.csv')

# 2. Prepare train-test split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2,
    stratify=y  # Important for imbalanced classes
)

# 3. MLflow setup
mlflow.set_experiment("Penguins-Klasifikasi")
# Set more conservative autologging to avoid version conflicts
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=True,
    silent=True
)

with mlflow.start_run():
    # 4. Model training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)