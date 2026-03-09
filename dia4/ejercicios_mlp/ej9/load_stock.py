import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

symbol = 'uber'
stock_df = pd.read_csv(
    'https://drive.upm.es/s/U9BQWmxAKNXorAB/download',
    parse_dates=['Date']
)
stock_df = stock_df.sort_values('Date')

# Create lag features for the past 5 days
for lag in range(1, 6):
    stock_df[f'lag_{lag}'] = stock_df['Close'].shift(lag)

stock_df = stock_df.dropna().reset_index(drop=True)

# Scale the data
scaler = MinMaxScaler()
cols_to_scale = ['Close'] + [f'lag_{i}' for i in range(1, 6)]
stock_df[cols_to_scale] = scaler.fit_transform(
    stock_df[cols_to_scale]
)

feature_cols = [f'lag_{i}' for i in range(1, 6)]
X = stock_df[feature_cols].values.astype(np.float32)
y = stock_df['Close'].values.astype(np.float32)

# Split by date (last 20% as test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Save dates for plotting
train_dates = stock_df['Date'][:train_size]
test_dates = stock_df['Date'][train_size:]

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test).view(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Symbol: {symbol}")
print(f"Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
