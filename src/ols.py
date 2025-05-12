# src/ols.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_ols(df, features, target='mag'):
    """
    Runs OLS regression and returns the fitted model.
    """
    df = df[df[target].notnull()].copy()
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df[target], errors='coerce').dropna()

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X

def compute_vif(X):
    """
    Computes Variance Inflation Factor (VIF) for each feature.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
