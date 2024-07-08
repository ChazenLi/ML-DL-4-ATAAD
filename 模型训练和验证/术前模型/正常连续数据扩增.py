import pandas as pd
import numpy as np

# 定义需要扩增的样本数量
n_samples = 1000

# 初始化数据字典
data = {
    'id': np.arange(302, 302 + n_samples),
    'gender': np.random.choice([1, 0], n_samples, p=[0.52, 0.48]),
    'smoking': [],
    'drinking': [],
    'hypertension': np.random.choice([1, 0], n_samples, p=[0.275, 1 - 0.275]),
    'diabetes': [],
    'COPD': [],
    'CAD': np.random.choice([1, 0], n_samples, p=[0.01, 1 - 0.01]),
    'stroke': np.zeros(n_samples, dtype=int),
    'cerebral_hemorrhage': np.zeros(n_samples, dtype=int),
    'Marfan_syndrome': np.random.choice([1, 0], n_samples, p=[1/5000, 1 - 1/5000]),
    'pericardial_effusion': np.zeros(n_samples, dtype=int),
    'age': [],
    'height': [],
    'weight': [],
    'creatinine': np.random.uniform(41, 81, n_samples),
    'EF': np.random.uniform(53, 73, n_samples),
    'lactate': np.random.uniform(0.5, 2.2, n_samples),
    'hemoglobin': np.random.uniform(115, 150, n_samples),
    'platelet': np.random.uniform(125, 350, n_samples),
    'blood_glucose': np.random.uniform(3.58, 6.05, n_samples),
    'BUN': np.random.uniform(2.86, 7.90, n_samples),
    'albumin': np.random.uniform(40, 55, n_samples)
}

# 吸烟、嗜酒、糖尿病、COPD 数据
for gender in data['gender']:
    if gender == 1:  # 男性
        data['smoking'].append(np.random.choice([1, 0], p=[0.521, 1 - 0.521]))
        data['drinking'].append(np.random.choice([1, 0], p=[0.0068, 1 - 0.0068]))
        data['diabetes'].append(np.random.choice([1, 0], p=[0.133, 1 - 0.133]))
        data['COPD'].append(np.random.choice([1, 0], p=[0.119, 1 - 0.119]))
    else:  # 女性
        data['smoking'].append(np.random.choice([1, 0], p=[0.025, 1 - 0.025]))
        data['drinking'].append(np.random.choice([1, 0], p=[0.00023, 1 - 0.00023]))
        data['diabetes'].append(np.random.choice([1, 0], p=[0.115, 1 - 0.115]))
        data['COPD'].append(np.random.choice([1, 0], p=[0.054, 1 - 0.054]))

# 年龄分布：60岁以上占13.5%，15-59岁占63.35%
age_distribution = [60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
age_prob = [0.135] + [0.6335 / (len(age_distribution) - 1)] * (len(age_distribution) - 1)
age_prob[0] = 1 - sum(age_prob[1:])
data['age'] = np.random.choice(age_distribution, n_samples, p=age_prob)

# 身高和体重分布
for gender in data['gender']:
    if gender == 1:  # 男性
        data['height'].append(np.random.normal(172.5, 7.5))  # 男性身高范围160-185
        data['weight'].append(np.random.normal(90, 15))      # 男性体重范围60-120
    else:  # 女性
        data['height'].append(np.random.normal(162.5, 7.5))  # 女性身高范围150-175
        data['weight'].append(np.random.normal(65, 7.5))     # 女性体重范围50-80

# 创建DataFrame
df_continuous = pd.DataFrame(data)

# 保存更新后的Excel文件
file_path_continuous = 'E:/Python-projects/pytorch/练习/项目实战/ATAAD疾病预测/术前预测模型/正常.xlsx'
df_continuous.to_excel(file_path_continuous, index=False)

file_path_continuous
