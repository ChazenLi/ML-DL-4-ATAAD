import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 读取Excel文件
file_path = r"E:\Python-projects\pytorch\练习\项目实战\ATAAD疾病预测\术前预测模型\完整数据.xlsx"
data = pd.read_excel(file_path)

# 提取特征和标签
X = data.iloc[:, 1:-1].values  # 除去第一列和最后一列，获取特征
y = data.iloc[:, -1].values    # 最后一列为标签（患病与否）

# 随机化数据的顺序
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将特征数据调整为 (batch_size, 1, 22, 1) 的形状
X_scaled = X_scaled.reshape(-1, 1, 22, 1)

# 划分训练集和测试集（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 使用SMOTE处理类别不平衡问题
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train.reshape(-1, 22), y_train)
X_train_balanced = X_train_balanced.reshape(-1, 1, 22, 1)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 11 * 1, 256)  # 调整全连接层的输入
        self.fc2 = nn.Linear(256, 2)  # 二分类问题

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 11 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = CNN()
class_weights = torch.tensor([1.0, 20.0], dtype=torch.float32)  # 根据样本不平衡程度调整权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练模型
num_epochs = 35
train_losses = []
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 绘制损失函数变化趋势
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('训练损失函数变化趋势')
plt.legend()
plt.show()

# 评估模型
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# 打印分类报告
print("Classification Report:")
print(classification_report(y_true, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["未患病", "患病"], yticklabels=["未患病", "患病"])
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 保存模型
model_path = r"E:\Python-projects\pytorch\练习\项目实战\ATAAD疾病预测\术前预测模型\cnn_model.pth"
torch.save(model.state_dict(), model_path)

print("模型保存成功！")
