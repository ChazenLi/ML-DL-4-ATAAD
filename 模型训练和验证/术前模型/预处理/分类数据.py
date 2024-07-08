import pandas as pd
import os

# 1. 读取数据
input_file_path = r"E:/dataset/ML4ATAAD/术前预测/正常人数据/正常人分类数据.xlsx"
df = pd.read_excel(input_file_path)

# 2. 数据清洗和填充缺失值（使用众数填充）
for column in df.columns:
    mode_value = df[column].mode()[0]  # 获取众数
    df[column] = df[column].fillna(mode_value)  # 用众数填充缺失值

# 3. One-hot编码（不包括编号列）
df = pd.get_dummies(df, drop_first=True)

# 4. 保存到新Excel文件
output_directory = r"E:/dataset/ML4ATAAD/术前预测/数据处理后"
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, "预处理后数据.xlsx")
df.to_excel(output_file_path, index=False)

print(f"预处理后的数据已保存到 {output_file_path}")
