import csv
import torch
import torch.nn as nn
import torch.optim as optim
from drag_get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data
x_2015, y_2015 = get_data(["2018"])

np.set_printoptions(suppress=True)

X_train, X_test, y_train, y_test = train_test_split(x_2015, y_2015, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Convert to torch tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the MLP model using PyTorch
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize the model
input_size = X_train_scaled.shape[1]
hidden_sizes = [128, 64, 4]  # Specify your hidden layers
output_size = 1
model = MLPModel(input_size, hidden_sizes, output_size)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 25000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_scaled)

    # Compute loss
    loss = loss_fn(y_pred, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Predictions
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_scaled)

# Convert predictions and true values to numpy for evaluation
y_pred_test = y_pred_test.numpy().flatten()
y_test = y_test.numpy().flatten()

# Calculate Mean Squared Error
mse = np.mean((y_pred_test - y_test) ** 2)
print(f"Mean Error: {np.sqrt(mse):.4f}")

errors = y_pred_test - y_test
mbe = np.mean(errors)

if mbe > 0:
    print(f"Model is overpredicting on average by {mbe:.2f} feet.")
elif mbe < 0:
    print(f"Model is underpredicting on average by {-mbe:.2f} feet.")
else:
    print("Model has no average bias.")

# Testing on test year data
x_test_year, y_test_year = get_data(["2019"])

test_mean_raw = np.mean(x_test_year, axis=0)
test_std_raw = np.std(x_test_year, axis=0)

print("Feature means (test set, raw):", test_mean_raw)
print("Feature stds (test set, raw):", test_std_raw)



x_test_year_scaled = scaler_x.transform(x_test_year)
x_test_year_scaled = torch.tensor(x_test_year_scaled, dtype=torch.float32)

# Model predictions on test set
with torch.no_grad():
    y_test_year_pred = model(x_test_year_scaled).numpy().flatten()

errors_test_year = y_test_year_pred - np.array(y_test_year)
mbe_test_year = np.mean(errors_test_year)

for i in range(len(errors_test_year)):
    print("Predicted:", round(y_test_year_pred[i], 3), "| Actual:", y_test_year[i])
    if y_test_year[i] > 150 and y_test_year[i] < 200:
        print("_________________")
if mbe_test_year > 0:
    print(f"Model is overpredicting on average by {mbe_test_year:.2f} feet.")
elif mbe_test_year < 0:
    print(f"Model is underpredicting on average by {-mbe_test_year:.2f} feet.")
else:
    print("Model has no average bias.")

# Bin analysis
y_test_year = np.array(y_test_year)
y_test_year_pred = model(torch.tensor(x_test_year_scaled, dtype=torch.float32)).detach().numpy().flatten()
errors_test_year = y_test_year_pred - y_test_year

bin_ranges = [(150, 200), (200, 250), (250, 300), (300, 350),(350,400),(400,450),(450,500)]

for (low, high) in bin_ranges:
    mask = (y_test_year >= low) & (y_test_year < high)
    subset_actual = y_test_year[mask]
    subset_pred = y_test_year_pred[mask]
    subset_errors = errors_test_year[mask]
    
    if len(subset_actual) == 0:
        print(f"No data points in range [{low}, {high})")
        print("-" * 50)
        continue
    
    avg_error = np.mean(subset_errors)
    over_pred_pct = np.mean(subset_errors > 0) * 100
    under_pred_pct = np.mean(subset_errors < 0) * 100
    
    print(f"Range [{low}, {high}):")
    print(f"  Number of samples: {len(subset_actual)}")
    print(f"  Average error (pred - actual): {avg_error:.2f}")
    print(f"  Over-prediction (%): {over_pred_pct:.1f}%")
    print(f"  Under-prediction (%): {under_pred_pct:.1f}%")
    print("-" * 50)


print(np.mean(y_test_year))
