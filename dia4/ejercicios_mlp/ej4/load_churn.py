import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

churn_df = pd.read_csv('https://drive.upm.es/s/qohWjbcn8jemNq7/download')
subset = churn_df[['gender', 'tenure', 'InternetService',
                    'MonthlyCharges', 'Churn']].copy()

# Encode gender: female->0, male->1
subset['gender'] = subset['gender'].map(
    {'Female': 0, 'Male': 1}
).astype(np.float32)
# One-hot encode InternetService
internet_dummies = pd.get_dummies(
    subset['InternetService'], prefix='Internet'
)
subset = pd.concat(
    [subset.drop('InternetService', axis=1), internet_dummies], axis=1
)
# Encode target
subset['Churn'] = subset['Churn'].map(
    {'Yes': 1, 'No': 0}
).astype(np.int64)

X = subset.drop(columns=['Churn']).values.astype(np.float32)
y = subset['Churn'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {list(subset.drop(columns=['Churn']).columns)}")
print(f"Churn rate: {y.mean():.2%}")
