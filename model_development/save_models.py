import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def save_models_from_existing():
    try:
        model = joblib.load('trained_models/best_model.pkl')
        scaler = joblib.load('trained_models/scaler.pkl')
        label_encoder = joblib.load('trained_models/label_encoder.pkl')
        feature_names = joblib.load('trained_models/feature_names.pkl')
        model_info = joblib.load('trained_models/model_info.pkl')
        
        print("Модели уже сохранены в папке 'trained_models/'")
        print(f"Тип модели: {model_info['model_type']}")
        print(f"Точность: {model_info['test_accuracy']:.4f}")
        
    except FileNotFoundError:
        print("Файлы моделей не найдены. Запустите сначала breast_cancer_analysis.py")

if __name__ == "__main__":
    save_models_from_existing()