import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

ins_df = pd.read_csv('https://drive.upm.es/s/HHiQmCuVk76jT0T/download')
ins_df['sex']    = ins_df['sex'].map({'female': 0, 'male': 1}).astype(np.float32)
ins_df['smoker'] = ins_df['smoker'].map({'no': 0, 'yes': 1}).astype(np.float32)
region_dummies = pd.get_dummies(ins_df['region'], prefix='region')
ins_df = pd.concat(
    [ins_df.drop('region', axis=1), region_dummies], axis=1
)

X = ins_df.drop('expenses', axis=1).values.astype(np.float32)
y = ins_df['expenses'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test).view(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Expenses range: [{y.min():.0f}, {y.max():.0f}] USD")
