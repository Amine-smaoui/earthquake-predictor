# src/svm.py

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score


def prepare_svm_data(combined):
    """
    Prepares features and labels for SVM.
    Returns: X_train, X_test, y_train, y_test (all scaled)
    """
    df = combined[combined['mag'].notnull()].copy()
    features = ['depth', 'dmin', 'latitude', 'longitude',
                'temperature', 'humidity', 'precipitation', 
                'sealevelPressure', 'surfacePressure', 'nst']
    X = df[features].fillna(0)
    y = (df['mag'] > 4.4).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


def train_svm_with_gridsearch(X_train, y_train):
    """
    Trains SVM with GridSearchCV over kernel types.
    Returns: grid_search object
    """
    param_grid = {'kernel': ['linear', 'rbf', 'poly']}
    grid_search = GridSearchCV(
        SVC(class_weight='balanced', random_state=42),
        param_grid,
        cv=5,
        scoring='balanced_accuracy',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_svm(grid_search, X_test, y_test):
    """
    Evaluates the best SVM model and returns predictions and score.
    """
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    score = balanced_accuracy_score(y_test, predictions)
    return predictions, score, best_model
