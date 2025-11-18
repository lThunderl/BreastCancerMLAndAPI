from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

model = None
scaler = None
label_encoder = None
feature_names = None
model_info = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global model, scaler, label_encoder, feature_names, model_info
    try:
        model = joblib.load('trained_models/best_model.pkl')
        scaler = joblib.load('trained_models/scaler.pkl')
        label_encoder = joblib.load('trained_models/label_encoder.pkl')
        feature_names = joblib.load('trained_models/feature_names.pkl')
        model_info = joblib.load('trained_models/model_info.pkl')
        
        print("✅ Все модели успешно загружены!")
        print(f"Тип модели: {model_info.get('model_type', 'Unknown')}")
        print(f"Точность: {model_info.get('test_accuracy', 'Unknown')}")
        print(f"Ожидаемые поля: {feature_names}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Проверьте наличие файлов в папке trained_models/:")
        import os
        if os.path.exists('trained_models'):
            files = os.listdir('trained_models')
            print("Найденные файлы:", files)
        else:
            print("Папка trained_models не существует")
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API для предсказания диагноза рака груди на основе характеристик опухоли",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    radius_mean: float = Field(..., description="Mean radius")
    texture_mean: float = Field(..., description="Mean texture")
    perimeter_mean: float = Field(..., description="Mean perimeter")
    area_mean: float = Field(..., description="Mean area")
    smoothness_mean: float = Field(..., description="Mean smoothness")
    compactness_mean: float = Field(..., description="Mean compactness")
    concavity_mean: float = Field(..., description="Mean concavity")
    concave_points_mean: float = Field(..., description="Mean concave points")
    symmetry_mean: float = Field(..., description="Mean symmetry")
    fractal_dimension_mean: float = Field(..., description="Mean fractal dimension")
    radius_se: float = Field(..., description="Radius standard error")
    texture_se: float = Field(..., description="Texture standard error")
    perimeter_se: float = Field(..., description="Perimeter standard error")
    area_se: float = Field(..., description="Area standard error")
    smoothness_se: float = Field(..., description="Smoothness standard error")
    compactness_se: float = Field(..., description="Compactness standard error")
    concavity_se: float = Field(..., description="Concavity standard error")
    concave_points_se: float = Field(..., description="Concave points standard error")
    symmetry_se: float = Field(..., description="Symmetry standard error")
    fractal_dimension_se: float = Field(..., description="Fractal dimension standard error")
    radius_worst: float = Field(..., description="Worst radius")
    texture_worst: float = Field(..., description="Worst texture")
    perimeter_worst: float = Field(..., description="Worst perimeter")
    area_worst: float = Field(..., description="Worst area")
    smoothness_worst: float = Field(..., description="Worst smoothness")
    compactness_worst: float = Field(..., description="Worst compactness")
    concavity_worst: float = Field(..., description="Worst concavity")
    concave_points_worst: float = Field(..., description="Worst concave points")
    symmetry_worst: float = Field(..., description="Worst symmetry")
    fractal_dimension_worst: float = Field(..., description="Worst fractal dimension")

class PredictionOutput(BaseModel):
    diagnosis: str
    prediction: int
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
async def root():
    return {
        "message": "Breast Cancer Prediction API", 
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    if all([model, scaler, label_encoder, feature_names]):
        return {
            "status": "healthy", 
            "models_loaded": True,
            "model_type": model_info.get('model_type', 'Unknown') if model_info else 'Unknown'
        }
    return {"status": "unhealthy", "models_loaded": False}

@app.get("/model-info")
async def get_model_info():
    if model_info:
        return model_info
    raise HTTPException(status_code=404, detail="Model info not available")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        if not all([model, scaler, feature_names]):
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        field_mapping = {
            'concave_points_mean': 'concave points_mean',
            'concave_points_se': 'concave points_se', 
            'concave_points_worst': 'concave points_worst'
        }
        
        input_df_renamed = input_df.rename(columns=field_mapping)
        
        input_df_renamed = input_df_renamed[feature_names]
        
        scaled_data = scaler.transform(input_df_renamed)
        
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return PredictionOutput(
            diagnosis=diagnosis,
            prediction=int(prediction),
            confidence=round(float(confidence), 4),
            probabilities={
                "benign": round(float(probability[0]), 4),
                "malignant": round(float(probability[1]), 4)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(input_data: List[PredictionInput]):
    try:
        predictions = []
        for item in input_data:
            input_dict = item.dict()
            input_df = pd.DataFrame([input_dict])
            
            # Переименовываем поля
            field_mapping = {
                'concave_points_mean': 'concave points_mean',
                'concave_points_se': 'concave points_se',
                'concave_points_worst': 'concave points_worst'
            }
            input_df_renamed = input_df.rename(columns=field_mapping)
            input_df_renamed = input_df_renamed[feature_names]
            
            scaled_data = scaler.transform(input_df_renamed)
            
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            
            diagnosis = "Malignant" if prediction == 1 else "Benign"
            confidence = probability[1] if prediction == 1 else probability[0]
            
            predictions.append({
                "diagnosis": diagnosis,
                "prediction": int(prediction),
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    "benign": round(float(probability[0]), 4),
                    "malignant": round(float(probability[1]), 4)
                }
            })
        
        return {
            "predictions": predictions,
            "total_count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)