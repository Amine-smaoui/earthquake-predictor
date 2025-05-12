# src/neural_net.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()                          # Activation function 
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
        self.sigmoid = nn.Sigmoid()                    # Output activation for binary classification

        
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

def prepare_nn_data(combined, features):
    X = combined[features].dropna()
    y = combined.loc[X.index, 'is_high_magnitude']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, scaler

# Trains and evaluates NN with different learning rates and hidden sizes
def train_and_evaluate_nn(X_train, y_train, X_val, y_val, learning_rates, hidden_sizes, epochs=300):
    best_score = 0
    best_model = None
    best_config = None
    results = []

    for lr in learning_rates:
        for hidden in hidden_sizes:
            model = SimpleClassifier(X_train.shape[1], hidden)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()

            for epoch in range(epochs):
                model.train()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                preds_binary = (val_preds > 0.5).int()
                acc = balanced_accuracy_score(y_val.numpy(), preds_binary.numpy())

            results.append((lr, hidden, acc))
            if acc > best_score:
                best_score = acc
                best_model = model
                best_config = (lr, hidden, acc)

    return best_model, best_config, results
