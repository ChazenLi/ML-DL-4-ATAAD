import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import seaborn as sns

# 设置中文字体和坐标轴负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
file_path = r"E:\Python-projects\pytorch\练习\项目实战\ATAAD疾病预测\术前预测模型\完整数据.xlsx"
data = pd.read_excel(file_path)

# 随机排序数据
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 提取特征和标签
X = data.iloc[:, 1:-1].values  # 除去第一列和最后一列，获取特征
y = data.iloc[:, -1].values    # 最后一列为标签（患病与否）

# 获取特征列名
feature_names = data.columns[1:-1]

# 将数据集划分为训练集和测试集（3:7）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 实例化随机森林分类器
rf_clf = RandomForestClassifier(random_state=42)

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'最佳参数: {grid_search.best_params_}')

# 使用最佳参数进行预测
best_rf_clf = grid_search.best_estimator_
y_pred = best_rf_clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy * 100:.2f}%')

# 打印分类报告
print(classification_report(y_test, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['未患病', '患病'], yticklabels=['未患病', '患病'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 获取特征重要性
feature_importances = best_rf_clf.feature_importances_

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), feature_importances, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('特征重要性')
plt.ylabel('特征')
plt.title('特征重要性分析')
plt.show()

# 对特征进行PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化PCA结果
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
for i, txt in enumerate(feature_names):
    plt.annotate(txt, (pca.components_[0, i], pca.components_[1, i]), fontsize=9, alpha=0.8, color='red')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA分析')
plt.colorbar(label='标签')
plt.show()

# 计算预测概率
y_proba = best_rf_clf.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.show()

# 绘制特征相关性的热图
plt.figure(figsize=(12, 10))
corr_matrix = data.iloc[:, 1:-1].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
plt.title('特征相关性热图')
plt.show()
