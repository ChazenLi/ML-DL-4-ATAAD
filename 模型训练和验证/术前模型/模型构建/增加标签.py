import pandas as pd

# 读取完整数据的Excel文件
file_path = r"E:\Python-projects\pytorch\练习\项目实战\ATAAD疾病预测\术前预测模型\完整数据.xlsx"
df = pd.read_excel(file_path)

# 增加一列“患病与否”
df['患病与否'] = 0
df.loc[:299, '患病与否'] = 1  # 前300个数据标记为1
df.loc[300:1299, '患病与否'] = 0  # 后1000个数据标记为0

# 保存修改后的数据回Excel文件
df.to_excel(file_path, index=False)

print("新增列“患病与否”并保存成功！")
