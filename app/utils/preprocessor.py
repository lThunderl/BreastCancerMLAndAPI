import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame):
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted yet")
        return self.scaler.transform(X)
    
    def save(self, filepath: str):
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.feature_names = data['feature_names']
        return preprocessor