import pandas as pd
import numpy as np

# 定义需要扩增的样本数量
n_samples = 1000

# 初始化数据字典
data = {
    'id': np.arange(302, 302 + n_samples),
    'gender': [],
    'smoking': [],
    'drinking': [],
    'hypertension': [],
    'diabetes': [],
    'COPD': [],
    'CAD': [],
    'stroke': np.zeros(n_samples, dtype=int),
    'cerebral_hemorrhage': np.zeros(n_samples, dtype=int),
    'Marfan_syndrome': [],
    'pericardial_effusion': np.zeros(n_samples, dtype=int)
}

# 正常人群性别比例
male_ratio = 0.52
female_ratio = 0.48

# 吸烟率
total_smoking_rate = 0.272
male_smoking_rate = 0.521
female_smoking_rate = 0.025

# 嗜酒率
male_drinking_rate = 0.0068
female_drinking_rate = 0.00023

# 高血压患病率
hypertension_rate = 0.275

# 糖尿病患病率
male_diabetes_rate = 0.133
female_diabetes_rate = 0.115
total_diabetes_rate = 0.124

# COPD患病率
male_COPD_rate = 0.119
female_COPD_rate = 0.054
total_COPD_rate = 0.086

# CAD患病率
CAD_rate = 0.01

# 马凡综合征患病率
Marfan_syndrome_rate = 1 / 5000

# 生成性别数据
data['gender'] = np.random.choice(['male', 'female'], n_samples, p=[male_ratio, female_ratio])

# 根据性别生成吸烟、嗜酒、糖尿病、COPD数据
for gender in data['gender']:
    if gender == 'male':
        data['smoking'].append(np.random.choice([1, 0], p=[male_smoking_rate, 1 - male_smoking_rate]))
        data['drinking'].append(np.random.choice([1, 0], p=[male_drinking_rate, 1 - male_drinking_rate]))
        data['diabetes'].append(np.random.choice([1, 0], p=[male_diabetes_rate, 1 - male_diabetes_rate]))
        data['COPD'].append(np.random.choice([1, 0], p=[male_COPD_rate, 1 - male_COPD_rate]))
    else:
        data['smoking'].append(np.random.choice([1, 0], p=[female_smoking_rate, 1 - female_smoking_rate]))
        data['drinking'].append(np.random.choice([1, 0], p=[female_drinking_rate, 1 - female_drinking_rate]))
        data['diabetes'].append(np.random.choice([1, 0], p=[female_diabetes_rate, 1 - female_diabetes_rate]))
        data['COPD'].append(np.random.choice([1, 0], p=[female_COPD_rate, 1 - female_COPD_rate]))

# 生成高血压数据
data['hypertension'] = np.random.choice([1, 0], n_samples, p=[hypertension_rate, 1 - hypertension_rate])

# 生成CAD数据
data['CAD'] = np.random.choice([1, 0], n_samples, p=[CAD_rate, 1 - CAD_rate])

# 生成马凡综合征数据
data['Marfan_syndrome'] = np.random.choice([1, 0], n_samples, p=[Marfan_syndrome_rate, 1 - Marfan_syndrome_rate])

# 更新性别列，使用0表示女性，1表示男性
data['gender'] = [1 if gender == 'male' else 0 for gender in data['gender']]

# 创建更新后的DataFrame
df_updated = pd.DataFrame(data)

# 保存更新后的Excel文件
file_path_updated = 'E:/Python-projects/pytorch/练习/项目实战/ATAAD疾病预测/术前预测模型/正常分类.xlsx'
df_updated.to_excel(file_path_updated, index=False)

file_path_updated

