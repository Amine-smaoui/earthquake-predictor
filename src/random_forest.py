# src/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

def prepare_rf_data(combined, features):
    """
    Prepares and scales data for Random Forest.
    Returns: X_train, X_val, y_train, y_val, feature_selector, scaler
    """
    X = combined[features].dropna()
    y = combined.loc[X.index, 'is_high_magnitude']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    base_rf.fit(X_train, y_train)

    selector = SelectFromModel(base_rf, prefit=True, threshold="median")
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)

    return X_train_sel, X_val_sel, y_train, y_val, selector, scaler


def train_random_forest(X_train, y_train):
    """
    Trains Random Forest with GridSearchCV.
    Returns: trained grid search object
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_random_forest(model, X_val, y_val):
    """
    Evaluates Random Forest model and returns predictions and score.
    """
    y_pred = model.predict(X_val)
    score = balanced_accuracy_score(y_val, y_pred)
    return y_pred, score
