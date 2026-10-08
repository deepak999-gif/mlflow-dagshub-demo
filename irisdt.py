import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Load Iris Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Hyperparameter Values
# --------------------------------------------------

max_depth =3
mlflow.set_experiment('decision_tree_experiment')
with mlflow.start_run(run_name='decision_tree_run'):
 # Create model
  model = DecisionTreeClassifier(
          max_depth=max_depth,
            random_state=42
        )

        # Train
  model.fit(X_train, y_train)

        # Prediction
  y_pred = model.predict(X_test)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
  accuracy = accuracy_score(y_test, y_pred)

  precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

  recall = recall_score(
            y_test,
            y_pred,
            average="weighted"
        )

  f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )
  mlflow.log_param("max_depth",max_depth)
  mlflow.log_metric("accuracy",accuracy)
  mlflow.log_metric("precision",precision)
  mlflow.log_metric('recall',recall)
  mlflow.log_metric('f1',f1)
  
  mlflow.sklearn.log_model(
    model,
    name="meral_dusra_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    
)
  mlflow.log_artifact("train.py")

print('Trained')

    