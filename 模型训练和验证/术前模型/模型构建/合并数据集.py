import pandas as pd

# 读取正常数据的Excel文件
file_path = r"E:\Python-projects\pytorch\练习\项目实战\ATAAD疾病预测\术前预测模型\正常.xlsx"
df = pd.read_excel(file_path)

# 将第一列的序号全部-1
df.iloc[:, 0] = df.iloc[:, 0] - 1

# 保存修改后的数据回Excel文件
df.to_excel(file_path, index=False)

print("序号已减1并保存成功！")
