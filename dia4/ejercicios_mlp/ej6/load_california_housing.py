import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch

housing = fetch_california_housing()
X = housing.data.astype(np.float32)
y = housing.target.astype(np.float32)  # median house value (100k USD)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test).view(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {housing.feature_names}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}] (×100k USD)")
