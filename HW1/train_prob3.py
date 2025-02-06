from robot_kinematics_regression import *
from hands_on_regression import *
import os
import sys

# Prepare the data:
train_data_path = os.path.join('./robot_kinematics_training_data.npz')
val_data_path = os.path.join('.//robot_kinematics_validation_data.npz')

train_data = np.load(train_data_path)
val_data = np.load(val_data_path)

# Unpack the data
# we will use x for input and y for target to keep consistent with previous question
x = torch.from_numpy(train_data['theta'])
y = torch.from_numpy(train_data['x'])

x_val = torch.from_numpy(val_data['theta'])
y_val = torch.from_numpy(val_data['x'])

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 生成数据
N = 1000  # 样本数量
theta_des = torch.rand(N, 3) * 2 * torch.pi - torch.pi  # 随机 3 维关节角 (-pi, pi)
x = torch.cat([torch.cos(theta_des[:, :1]) + torch.sin(theta_des[:, 1:2]),
               torch.sin(theta_des[:, :1]) - torch.cos(theta_des[:, 1:2])], dim=1)  # 模拟正运动学

# 归一化
mean_theta = theta_des.mean(dim=0)
std_theta = theta_des.std(dim=0)
mean_x = x.mean(dim=0)
std_x = x.std(dim=0)

theta_des_norm = (theta_des - mean_theta) / std_theta
x_norm = (x - mean_x) / std_x

# 构造 DataLoader
dataset = TensorDataset(theta_des_norm, x_norm)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

num_epochs = 200
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    total_loss = 0
    for batch_theta, batch_x in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_theta)
        loss = loss_fn(y_pred, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss / len(train_loader):.6f}')

# **保存模型**
torch.save(model.state_dict(), "robot_kinematics_model.pt")

model.eval()
with torch.no_grad():
    y_pred_norm_all = model(theta_des_norm)
    y_pred_all = y_pred_norm_all * std_x + mean_x

    mse_on_original_scale = loss_fn(y_pred_all, x).item()

print(f"Final MSE on original scale: {mse_on_original_scale:.6f}")

norm_constants = {
    "theta_mean": mean_theta,
    "theta_std": std_theta,
    "x_mean": mean_x,
    "x_std": std_x
}
torch.save(norm_constants, "robot_kinematics_norm_constants.pt")