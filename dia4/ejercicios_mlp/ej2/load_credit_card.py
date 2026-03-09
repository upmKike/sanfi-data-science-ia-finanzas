import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch

# Load the credit card default dataset
df = pd.read_csv('https://drive.upm.es/s/u8juq4XZyElqiAs/download')
selected_cols = [
    'LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1',
    'PAY_AMT1', 'default.payment.next.month'
]
data = df[selected_cols]
data = data.apply(pd.to_numeric, errors='coerce').dropna()

X = data.drop(columns=['default.payment.next.month']).values.astype(np.float32)
y = data['default.payment.next.month'].values.astype(np.float32)

# Normalize features (mean=0, std=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Correct unbalanced dataset with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split and convert to tensors
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Distribución original: {pd.Series(y).value_counts().to_dict()}")
