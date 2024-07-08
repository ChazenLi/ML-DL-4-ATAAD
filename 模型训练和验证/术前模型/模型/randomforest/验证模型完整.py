import pickle

model_path = "E:\\Python-projects\\best_rf_model.pkl"

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("模型加载成功")
except Exception as e:
    print(f"加载模型失败: {e}")
