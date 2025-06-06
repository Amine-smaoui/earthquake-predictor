# Earthquake Prediction Project

## Overview

This project implements various machine learning models to predict high-magnitude earthquakes using seismic and weather data. The system combines earthquake event data with weather conditions to improve prediction accuracy.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── utils.py              # Data preparation and utility functions
│   ├── ols.py               # Ordinary Least Squares regression
│   ├── pcr.py               # Principal Component Regression
│   ├── logisticRegression.py # Logistic Regression models
│   ├── svm.py               # Support Vector Machine implementation
│   ├── neural_net.py        # Neural Network model
│   ├── random_forest.py     # Random Forest classifier
│   └── unsupervised_learning.py # Clustering analysis
├── data/                    # Data directory for CSV files
├── .gitignore              # Git ignore configuration
└── README.md               # This file
```

## Features

- Multiple machine learning models:
  - Logistic Regression (with L1 regularization)
  - Support Vector Machines (SVM)
  - Neural Networks
  - Random Forest
  - Principal Component Regression (PCR)
  - Ordinary Least Squares (OLS)
- Unsupervised learning for pattern discovery
- Comprehensive data preprocessing pipeline
- Weather data integration
- Model evaluation and comparison

## Installation

1. Clone the repository:

```bash
git clone https://github.com/smaoui-me/earthquake-predictor.git
```

2. Create and activate a virtual environment:

```bash
python -m venv env
.\env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The project uses two main data sources:

1. Earthquake events data
2. Weather data

Data preprocessing includes:

- Time series conversion
- Feature extraction
- Data cleaning
- Weather data aggregation
- Merging of earthquake and weather data

## Model Implementation

### 1. Logistic Regression

- Standard implementation
- L1 regularization variant
- Includes data scaling and preprocessing

### 2. Support Vector Machine (SVM)

- Multiple kernel options (linear, RBF, polynomial)
- Grid search for hyperparameter optimization
- Balanced class weights

### 3. Neural Network

- Simple feed-forward architecture
- Configurable hidden layer sizes
- Adam optimizer with BCE loss

### 4. Random Forest

- Feature selection
- Grid search for optimal parameters
- Balanced accuracy scoring

### 5. Principal Component Regression

- PCA for dimensionality reduction
- Combined with logistic regression
- Standardized pipeline

### 6. Unsupervised Learning

- K-means clustering
- Feature-based earthquake pattern analysis

## Usage

1. Prepare your data:

```python
from src.utils import prepare_events, prepare_weather, merge_data

events = prepare_events('path/to/events.csv')
weather = prepare_weather('path/to/weather.csv')
combined_data = merge_data(events, weather)
```

2. Train and evaluate models:

```python
from src.logisticRegression import prepare_train_test_sets, train_logistic
from src.svm import prepare_svm_data, train_svm_with_gridsearch
# ... import other models as needed

# Example with logistic regression
X_tr, X_val, y_tr, y_val, X_test = prepare_train_test_sets(combined_data, features)
model = train_logistic(X_tr, y_tr)
```

## Model Evaluation

Each model includes evaluation functions that return:

- Predictions
- Balanced accuracy scores
- Model-specific metrics

---

**Note**: This project is designed for research and educational purposes. The predictions should not be used as the sole basis for earthquake-related decision-making.
