# src/pcr.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

def train_pcr(X_train, y_train):
    """
    Trains a pipeline with StandardScaler, PCA, and LogisticRegression.
    Returns the trained pipeline.
    """
    pcr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('logreg', LogisticRegression(max_iter=1000))
    ])
    pcr_pipeline.fit(X_train, y_train)
    return pcr_pipeline

def evaluate_pcr(pipeline, X_val, y_val):
    """
    Evaluates PCR model and returns predictions, score, and the PCA object.
    """
    val_preds = pipeline.predict(X_val)
    score = balanced_accuracy_score(y_val, val_preds)
    pca = pipeline.named_steps['pca']
    return val_preds, score, pca
