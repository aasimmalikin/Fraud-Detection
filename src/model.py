import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from config import Model_DIR
import joblib

def train_model(X_train, y_train, X_val, y_val, params: dict):
    with mlflow.start_run():
        mlflow.log_params(params)
        model = XGBClassifier(**params, random_state=42, scale_pos_weight = (y_train==0).sum()/(y_train==1).sum())
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose = 100)
        val_preds = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, val_preds)
        mlflow.log_metric("val_auc", auc)
        mlflow.sklearn.log_model(model, "xgb_model")
        print(classification_report(y_val, (val_preds>0.5).astype(int)))
    return model

def save_model(model, path = None):
    path = path or Model_DIR / "xgb_model.pkl"
    joblib.dump(model, path)

def load_model(path = None):
    path = path or Model_DIR / "xgb_model.pkl"
    return joblib.load(path)


