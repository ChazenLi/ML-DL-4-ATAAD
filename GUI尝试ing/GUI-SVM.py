import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 预定义特征名称
numerical_features = ["年龄", "身高厘米", "体重（kg）", "肌酐", "EF（%）", "乳酸（mmol/L）", 
                      "血红蛋白（g/l）", "血小板（*109）", "血糖（mmol/L）", "BUN（mmol/l）", "白蛋白（g/l）"]
categorical_features = ["性别（男=1，女=0）", "吸烟史（有=1）", "嗜酒（有=1）", "高血压（有=1）", "糖尿病（有=1）", 
                        "COPD（有=1）", "CAD（有=1）", "脑卒中（有=1）", "脑出血（有=1）", "马凡综合征（有=1）", 
                        "心包积液（无=0，少量=1，中量=2，大量=3）"]

# 加载模型
model_path = "E:\\Python-projects\\best_svm_model.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    raise ValueError(f"加载模型失败: {e}")

# GUI应用
class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("疾病预测")

        self.entries = {}
        for feature in numerical_features + categorical_features:
            self.add_input_field(feature)

        self.submit_button = tk.Button(root, text="提交", command=self.submit)
        self.submit_button.pack()

    def add_input_field(self, feature):
        frame = tk.Frame(self.root)
        label = tk.Label(frame, text=feature)
        label.pack(side="left")
        entry = tk.Entry(frame)
        entry.pack(side="right")
        frame.pack()
        self.entries[feature] = entry

    def submit(self):
        try:
            input_data = {}
            for feature in numerical_features + categorical_features:
                value = self.entries[feature].get()
                if feature in numerical_features:
                    input_data[feature] = float(value)
                else:
                    input_data[feature] = int(value)

            # 数据预处理
            df = pd.DataFrame([input_data])
            preprocessed_data = self.preprocess(df)

            # 预测
            prediction = model.predict(preprocessed_data)
            messagebox.showinfo("预测结果", f"预测的疾病风险: {prediction[0]}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def preprocess(self, df):
        try:
            # 定义预处理步骤
            numerical_transformer = MinMaxScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            # 应用预处理步骤
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            preprocessed_data = pipeline.fit_transform(df)

            return preprocessed_data
        except Exception as e:
            raise ValueError("数据预处理失败: " + str(e))

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictionApp(root)
    root.mainloop()
