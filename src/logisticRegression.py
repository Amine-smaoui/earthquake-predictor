# src/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_train_test_sets(df, features, label='is_high_magnitude'):
    """
    Splits data into train/validation and test sets, fills NaNs using median.
    """
    train_data = df[df['mag'].notna()].copy()
    test_data = df[df['id'].notna()].copy()

    X_train = train_data[features].fillna(train_data[features].median())
    y_train = train_data[label]

    X_test = test_data[features].fillna(X_train.median())  # Use train median to fill

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_tr, X_val, y_tr, y_val, X_test


def train_logistic(X_tr, y_tr, max_iter=1000):
    """
    Trains a standard logistic regression model.
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_tr, y_tr)
    return model


def train_l1_logistic(X_tr, y_tr, C=1.0, max_iter=1000):
    """
    Trains a logistic regression model with L1 regularization.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=max_iter))
    ])
    pipeline.fit(X_tr, y_tr)
    return pipeline
