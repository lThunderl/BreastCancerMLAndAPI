О Проекте:
Этот проект представляет собой полный цикл разработки ML-решения:

Анализ данных и предобработка

Обучение модели логистической регрессии

REST API на FastAPI для предсказаний

Документация и тестирование

Модель: Logistic Regression с оптимизацией гиперпараметров

Данные: 30 медицинских признаков опухолей

Модель достигает точности ~97% в предсказании злокачественных (M) и доброкачественных (B) опухолей.

Технологии
Python, Scikit-learn, Pandas, NumPy

FastAPI, Uvicorn

Joblib для сериализации моделей

Matplotlib/Seaborn для визуализации

Быстрый старт:

pip install -r requirements.txt

cd breast_cancer_api
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

API будет доступно по адресу: http://localhost:8000/docs

Для запуска тестов на предсказания: 

cd tests
py test_api.py