import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
input_file_path = r"E:\dataset\ML4ATAAD\术前预测\正常人数据\正常人连续数据.xlsx"
df = pd.read_excel(input_file_path)

# 2. 数据清洗和填充缺失值（使用平均值填充）
for column in df.columns[1:]:  # 跳过第一列
    mean_value = df[column].mean()  # 获取均值
    df[column] = df[column].astype(float)  # 将列转换为浮点数类型
    df[column].fillna(mean_value, inplace=True)  # 用均值填充缺失值

# 3. 标准化操作（跳过第一列）
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# 4. 保存到新Excel文件
output_directory = r"E:\dataset\ML4ATAAD\术前预测\数据处理后"
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, "标准化预处理后连续数据.xlsx")
df.to_excel(output_file_path, index=False)

print(f"预处理后的数据已保存到 {output_file_path}")
