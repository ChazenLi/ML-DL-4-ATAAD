import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 读取数据
input_file_path = r"E:\dataset\ML4ATDD\原始xlsx文件\术中连续数据.xlsx"
df = pd.read_excel(input_file_path)

# 2. 数据清洗和填充缺失值（使用平均值填充）
for column in df.columns[1:]:  # 跳过第一列
    mean_value = df[column].mean()  # 获取均值
    df[column].fillna(mean_value, inplace=True)  # 用均值填充缺失值

# 3. 归一化操作（跳过第一列）
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# 4. 保存到新Excel文件
output_directory = r"E:\dataset\ML4ATDD\预处理后"
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, "预处理后连续数据.xlsx")
df.to_excel(output_file_path, index=False)

print(f"预处理后的数据已保存到 {output_file_path}")

