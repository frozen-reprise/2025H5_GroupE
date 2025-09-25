import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.stattools import adfuller  # For stationarity check, like in ARIMA notebook
import random


# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to master file (same as in ARIMA notebook)
data_filename = 'Final_master_daily_with_env.csv'
data_dir = Path(__file__).resolve().parent / 'data' / 'clean_datasets'
file_path = data_dir / data_filename  # Replace with your file


# Load data (adapted: daily aggregated, parse 'DATE', no resampling)
df = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
df = df.sort_index()  # Ensure chronological order

# Select target (adapted: 'demand_mw_mean' instead of 'demand_mw')
target = 'demand_mw_mean'

# Exclude leakage columns (concurrent targets, forecasts, and derived errors)
exclude_cols = ['demand_mw_sum', 'demand_mw_mean', 'demand_mw_max', 'forecast_mw_sum', 'forecast_mw_mean', 'forecast_mw_max', 'fcst_error_mean', 'fcst_abs_error_sum', 'fcst_ape_mean']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Handle missing values: drop rows with NaN for simplicity
df = df.dropna()

# Extract X (features) and y (target)
X = df[feature_cols].values
y = df[target].values

# Stationarity check on target (for reference, like in ARIMA)
adf_result = adfuller(y)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
if adf_result[1] > 0.05:
	print("Series is non-stationary (as in ARIMA, may need differencing internally in NN).")

# Train/test split (match ARIMA: last 365 days as test)
train_size = len(y) - 365
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale data
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train.reshape(-1, 1))

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Custom Dataset class for PyTorch
class TimeSeriesDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

# Create full train dataset and loader
train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation split (20% of train)
val_size = int(0.2 * len(train_dataset))
train_size_loader = len(train_dataset) - val_size
train_dataset_loader, val_dataset = torch.utils.data.random_split(train_dataset, [train_size_loader, val_size])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model 1: Fully Connected Network (MLP)
class FCModel(nn.Module):
	def __init__(self, input_size):
		super(FCModel, self).__init__()
		self.fc1 = nn.Linear(input_size, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Input size adapted to number of features
num_features = X_train.shape[1]
model_fc = FCModel(num_features).to(device)
optimizer_fc = optim.Adam(model_fc.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training function (same as before)
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=200, patience=10):
	best_loss = float('inf')
	patience_counter = 0
	for epoch in range(epochs):
		model.train()
		train_loss = 0
		for X_batch, y_batch in train_loader:
			X_batch, y_batch = X_batch.to(device), y_batch.to(device)
			optimizer.zero_grad()
			output = model(X_batch)
			loss = criterion(output.squeeze(), y_batch)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss /= len(train_loader)

		model.eval()
		val_loss = 0
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch, y_batch = X_batch.to(device), y_batch.to(device)
				output = model(X_batch)
				val_loss += criterion(output.squeeze(), y_batch).item()
		val_loss /= len(val_loader)

		print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

		# Early stopping
		if val_loss < best_loss:
			best_loss = val_loss
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print("Early stopping triggered.")
				break

print("Training Fully Connected Network...")
train_model(model_fc, train_loader, val_loader, optimizer_fc, criterion)

# Model 2: 1D CNN + FC (adapted: treats features as 1D sequence)
class CNNModel(nn.Module):
	def __init__(self, input_length):
		super(CNNModel, self).__init__()
		self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool1d(2)
		self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
		# Flatten size: after conv1 (same length), pool (//2), conv2 (same)
		self.flatten_size = 64 * (input_length//2)
		self.fc1 = nn.Linear(self.flatten_size, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, x):
		x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
		x = torch.relu(self.conv1(x))
		x = self.pool(x)
		x = torch.relu(self.conv2(x))
		x = x.view(x.size(0), -1)  # Flatten
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model_cnn = CNNModel(num_features).to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

print("Training 1D CNN + FC Network...")
train_model(model_cnn, train_loader, val_loader, optimizer_cnn, criterion)

# Function for prediction (adapted: direct prediction on test features, no recursive since features available)
def predict(model, X_scaled, scaler_y, device):
	model.eval()
	with torch.no_grad():
		X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
		pred_scaled = model(X_tensor).cpu().numpy().squeeze()
		pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
	return pred

# Predict on test set
pred_fc = predict(model_fc, X_test_scaled, scaler_y, device)
pred_cnn = predict(model_cnn, X_test_scaled, scaler_y, device)

# Evaluate
mae_fc = mean_absolute_error(y_test, pred_fc)
rmse_fc = np.sqrt(mean_squared_error(y_test, pred_fc))
print(f"Fully Connected - MAE: {mae_fc}, RMSE: {rmse_fc}")

mae_cnn = mean_absolute_error(y_test, pred_cnn)
rmse_cnn = np.sqrt(mean_squared_error(y_test, pred_cnn))
print(f"1D CNN + FC - MAE: {mae_cnn}, RMSE: {rmse_cnn}")

# Plots (match ARIMA: actual vs predicted, with 30-day smoothing)
dates = df.index[-len(y_test):]  # Dates for test period

plt.figure(figsize=(20, 6))
plt.plot(dates, y_test, label='Actual')
plt.plot(dates, pred_fc, label='FC Predicted')
plt.plot(dates, pred_cnn, label='CNN Predicted')
plt.legend()
plt.title('Actual vs Predicted Demand')
plt.show()

# Smoothed (30-day rolling mean, like in ARIMA)
actual_smooth = pd.Series(y_test).rolling(30).mean()
pred_fc_smooth = pd.Series(pred_fc).rolling(30).mean()
pred_cnn_smooth = pd.Series(pred_cnn).rolling(30).mean()

plt.figure(figsize=(20, 6))
plt.plot(dates, actual_smooth, label='Actual (Smoothed)')
plt.plot(dates, pred_fc_smooth, label='FC Predicted (Smoothed)')
plt.plot(dates, pred_cnn_smooth, label='CNN Predicted (Smoothed)')
plt.legend()
plt.title('Smoothed Actual vs Predicted Demand')
plt.show()

a = input("Pausing to keep windows open")