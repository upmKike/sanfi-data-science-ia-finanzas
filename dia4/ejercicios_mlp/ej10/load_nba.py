import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

nba_df = pd.read_csv(
    'https://drive.upm.es/s/NenPBTtldSDKLxQ/download'
)
subset = nba_df[['PPG', 'APG', 'RPG', 'Age', 'Pos', 'Salary']].copy()

# One-hot encode position
pos_dummies = pd.get_dummies(subset['Pos'], prefix='Pos')
subset = pd.concat(
    [subset.drop('Pos', axis=1), pos_dummies], axis=1
)

X = subset.drop('Salary', axis=1).values.astype(np.float32)
y = subset['Salary'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test).view(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Positions: {nba_df['Pos'].unique()}")
print(f"Salary range: [{y.min():.0f}, {y.max():.0f}] USD")
