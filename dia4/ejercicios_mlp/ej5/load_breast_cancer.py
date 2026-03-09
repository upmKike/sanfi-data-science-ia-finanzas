import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

cancer = load_breast_cancer()
X = cancer.data.astype(np.float32)    # shape (569, 30 features)
y = cancer.target.astype(np.int64)    # 0 = benign, 1 = malignant

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {cancer.feature_names[:5]}... ({len(cancer.feature_names)} total)")
print(f"Malignant rate: {(y == 1).mean():.2%}")
