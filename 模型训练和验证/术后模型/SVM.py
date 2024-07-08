import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 读取Excel文件
file_path = r"E:\dataset\ML4ATAAD\预处理后\one-hot+归一化\术中数据.xlsx"
data = pd.read_excel(file_path)

# 打印数据的前几行，确保数据读取正确
print(data.head())

# 提取特征和标签
X = data.iloc[:, 1:65]  # 第2到第65列为特征
y = data.iloc[:, 65]    # 第66列为标签（死亡与否的结果）

# 随机处理数据的顺序
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 使用SMOTE处理类别不平衡问题
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 使用支持向量机模型
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']
}

# 网格搜索找到最佳参数
svc = SVC()
svc_gscv = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
svc_gscv.fit(X_train_balanced, y_train_balanced)

# 输出最佳参数
print(f"Best Parameters: {svc_gscv.best_params_}")

# 使用最佳参数进行预测
best_svc = svc_gscv.best_estimator_
y_pred = best_svc.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印详细的分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["未死亡", "死亡"], yticklabels=["未死亡", "死亡"])
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 特征相关性分析
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X).corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("特征相关性矩阵")
plt.show()

# 特征贡献度分析（仅适用于线性核）
if best_svc.kernel == 'linear':
    feature_importance = best_svc.coef_.ravel()
    feature_importance_df = pd.DataFrame({
        'Feature': data.columns[1:65],
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("特征贡献度")
    plt.show()
else:
    print("非线性核不支持特征重要性分析")
