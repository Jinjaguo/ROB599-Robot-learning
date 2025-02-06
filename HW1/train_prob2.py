from hands_on_regression import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys


class SimpleDataset(Dataset):
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        x_i = self.Xs[idx]
        y_i = self.ys[idx]
        return x_i, y_i


# #xpand data using the polynomial basis fuctions
degree = 6  # Minimum degree that explains the data.
Xs_polynomial = polynomial_basis_functions(xs, degree)
Xs_polynomial_val = polynomial_basis_functions(xs_val, degree)

# Normalize the data
X_mean, X_std = get_normalization_constants(Xs_polynomial)
y_mean, y_std = get_normalization_constants(ys)
Xs_polynomial_norm = normalize_tensor(Xs_polynomial, X_mean, X_std)
ys_polynomial_norm = normalize_tensor(ys, y_mean, y_std)
Xs_polynomial_val_norm = normalize_tensor(Xs_polynomial_val, X_mean, X_std)
ys_polynomial_val_norm = normalize_tensor(ys_val, y_mean, y_std)

# Force Xs to have the first column as 1s to have the bias effect and avoid normalization
Xs_polynomial_norm[:, 0] = 1  # Force biases to 1
Xs_polynomial_val_norm[:, 0] = 1  # Force biases to 1

# Create Datasets
train_dataset = SimpleDataset(Xs_polynomial_norm, ys_polynomial_norm)
val_dataset = SimpleDataset(Xs_polynomial_val_norm, ys_polynomial_val_norm)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

# save normalization constants:
norm_constants = {
   'X_mean': X_mean,
   'X_std': X_std,
   'y_mean': y_mean,
   'y_std': y_std,
}
save_path = os.path.join('./regression_norm_constants.pt')
torch.save(norm_constants, save_path)

LR = 0.05
NUM_EPOCHS = 20000

linear_regressor = LinearRegressor(Xs_polynomial.shape[-1])

train_losses, val_losses = train_model(linear_regressor, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR)

# plot train loss and test loss:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
axes[0].plot(train_losses)
axes[0].grid()
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[0].set_yscale('log')
axes[1].plot(val_losses)
axes[1].grid()
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Loss')
axes[1].set_yscale('log')

# save model:
save_path = os.path.join('./linear_regressor.pt')
torch.save(linear_regressor.state_dict(), save_path)

# create evaluation data:
x_nn = torch.linspace(-5, 5, 1000, dtype=torch.float32)[:,None]
X_nn = polynomial_basis_functions(x_nn, degree)
# Normalize data
X_nn_norm = normalize_tensor(X_nn, X_mean, X_std )
X_nn_norm[:,0] = 1 # Force biases to 1

y_nn_norm = linear_regressor(X_nn_norm)
y_nn = denormalize_tensor(y_nn_norm, y_mean, y_std).detach().cpu()
plt.scatter(xs, ys, color='b', label='train_data')
plt.scatter(xs_val, ys_val, color='g', label='val_data')
plt.plot(x_nn, y_nn, color='orange', label='NN regression solution')
plt.grid()
plt.legend()
plt.show()

y_pred_norm = linear_regressor(Xs_polynomial_norm)
y_pred = denormalize_tensor(y_pred_norm, y_mean, y_std)
train_score = F.mse_loss(y_pred, ys).item()
y_pred_norm_val = linear_regressor(Xs_polynomial_val_norm)
y_pred_val = denormalize_tensor(y_pred_norm_val, y_mean, y_std)
val_score = F.mse_loss(y_pred_val, ys_val).item()
print(f'Train set score: {train_score}')
print(f'Validation set score: {val_score}')

# create evaluation data:
x_nn = torch.linspace(-5, 5, 1000, dtype=torch.float32)[:,None]
X_nn = polynomial_basis_functions(x_nn, degree)
# Normalize data
X_nn_norm = normalize_tensor(X_nn, X_mean, X_std )
X_nn_norm[:,0] = 1 # Force biases to 1

y_nn_norm = linear_regressor(X_nn_norm)
y_nn = denormalize_tensor(y_nn_norm, y_mean, y_std).detach().cpu()
plt.scatter(xs, ys, color='b', label='train_data')
plt.scatter(xs_val, ys_val, color='g', label='val_data')
plt.plot(x_nn, y_nn, color='orange', label='NN regression solution')
plt.grid()
plt.legend()
plt.show()