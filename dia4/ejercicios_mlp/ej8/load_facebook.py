import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

fb_df = pd.read_csv(
    'https://drive.upm.es/s/rADZBPmngfUJCPG/download', sep=';'
)

features = fb_df[['Page total likes', 'Type', 'Category',
                   'Post Month', 'Post Weekday', 'Post Hour',
                   'Paid']].copy()
target = fb_df['comment']

# Remove NaN
mask = ~(features.isna().any(axis=1) | target.isna())
features = features[mask]
target = target[mask]

# One-hot encode Type and Category
type_dummies = pd.get_dummies(features['Type'], prefix='Type')
cat_dummies  = pd.get_dummies(
    features['Category'].astype(str), prefix='Category'
)
X_df = pd.concat(
    [features.drop(['Type', 'Category'], axis=1),
     type_dummies, cat_dummies], axis=1
)
y = target.values.astype(np.float32)
X = X_df.values.astype(np.float32)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test).view(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Comments range: [{y.min():.0f}, {y.max():.0f}]")
