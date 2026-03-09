import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch

# Load Iris dataset
iris = load_iris()
X = iris.data.astype(np.float32)           # shape (150, 4)
y = iris.target.astype(np.int64)            # labels 0,1,2 for the three species

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Clases: {np.unique(y)}")
