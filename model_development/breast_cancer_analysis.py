import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

df = pd.read_csv('data.csv')

print("Общая информация о данных: \n")
print(df.shape)
df.info()
print("\nРаспределение целевой переменной:")
print(df['diagnosis'].value_counts())

print("\nКоличество нулевых значений:")
print(df.isnull().sum().sum())

plt.figure(figsize=(8,6))
sns.countplot(x='diagnosis', data=df)
plt.title('Распределение целевой переменной')
plt.savefig('target_distribution.png')
plt.show()

df = df.drop('id', axis=1)
df = df.drop_duplicates()
print(f"\nДанные после удаления дубликатов: {df.shape}")

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df[numerical_cols].hist(bins=20, figsize=(20, 15), layout=(5, 6))
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.show()

plt.figure(figsize=(20,15))
for i, col in enumerate(numerical_cols):
    plt.subplot(5, 6, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()

def remove_outliers_iqr(df, cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[cols] < lower_bound) | (df[cols] > upper_bound)).any(axis=1)]

initial_size = len(df)
df = remove_outliers_iqr(df, numerical_cols)
final_size = len(df)
print(f"\nУдалено выбросов: {initial_size - final_size}")
print(f"Осталось наблюдений: {final_size}")

plt.figure(figsize=(16,12))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
plt.savefig('correlation_matrix.png')
plt.show()

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("БАЗОВАЯ МОДЕЛЬ")
print("="*50)

logreg = LogisticRegression(
   max_iter=1000,
   random_state=42,
   class_weight='balanced',
   solver='liblinear'
)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
logreg_matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок базовой модели:")
print(logreg_matrix)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, 
                                      xticks_rotation='vertical', 
                                      im_kw={'cmap': 'viridis'})
plt.title('Матрица ошибок - Базовая модель')
plt.savefig('confusion_matrix_baseline.png')
plt.show()

print("Classification Report базовой модели:")
print(classification_report(y_test, y_pred))

print("Roc-curve - базовой модели:")
y_pred_proba = logreg.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Данные 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

print("\n" + "="*50)
print("КРОСС-ВАЛИДАЦИЯ")
print("="*50)

logreg_cv = LogisticRegression(
   max_iter=1000,
   random_state=42,
   class_weight='balanced',
   solver='liblinear'
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg_cv, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print("Результаты по фолдам: ", [f"{score:.4f}" for score in cv_scores])
print(f"Средняя точность: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n" + "="*50)
print("НАСТРОЙКА ГИПЕРПАРАМЕТРОВ")
print("="*50)

param_grid = {
   'C': [0.1, 1, 10, 100],
   'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}],
   'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
   LogisticRegression(max_iter=5000, random_state=42),
   param_grid,
   cv=5,
   scoring="accuracy",
   n_jobs=1,
   verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print(f"Лучшая точность на CV: {grid_search.best_score_:.4f}")

best_logreg = grid_search.best_estimator_
best_logreg.fit(X_train_scaled, y_train)

y_pred_best = best_logreg.predict(X_test_scaled)
best_matrix = confusion_matrix(y_test, y_pred_best)
print("\nМатрица ошибок лучшей модели:")
print(best_matrix)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, colorbar=False, 
                                      xticks_rotation='vertical', 
                                      im_kw={'cmap': 'viridis'})
plt.title('Матрица ошибок - Лучшая модель')
plt.savefig('confusion_matrix_best.png')
plt.show()

print("Classification Report - лучшей модели:")
print(classification_report(y_test, y_pred_best))

print("Roc-curve - лучшей модели: ")
y_pred_proba = best_logreg.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Данные 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

print("\n" + "="*50)
print("СОХРАНЕНИЕ МОДЕЛЕЙ")
print("="*50)

os.makedirs('trained_models', exist_ok=True)

joblib.dump(best_logreg, 'trained_models/best_model.pkl')
joblib.dump(scaler, 'trained_models/scaler.pkl')
joblib.dump(le, 'trained_models/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'trained_models/feature_names.pkl')

model_info = {
    'model_type': 'LogisticRegression',
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_accuracy': best_logreg.score(X_test_scaled, y_test),
    'feature_names': X.columns.tolist(),
    'classes': le.classes_.tolist()
}

joblib.dump(model_info, 'trained_models/model_info.pkl')

print("Модели и препроцессоры успешно сохранены в папку 'trained_models/'")
print(f"Точность на тестовых данных: {best_logreg.score(X_test_scaled, y_test):.4f}")

print("\n" + "="*50)
print("СВОДКА РЕЗУЛЬТАТОВ")
print("="*50)
print(f"Исходный размер данных: {pd.read_csv('data.csv').shape}")
print(f"Размер после предобработки: {df.shape}")
print(f"Лучшие параметры модели: {grid_search.best_params_}")
print(f"Точность на кросс-валидации: {grid_search.best_score_:.4f}")
print(f"Точность на тестовых данных: {best_logreg.score(X_test_scaled, y_test):.4f}")
print(f"Сохраненные файлы:")
print("  - trained_models/best_model.pkl (модель)")
print("  - trained_models/scaler.pkl (скейлер)")
print("  - trained_models/label_encoder.pkl (кодировщик)")
print("  - trained_models/feature_names.pkl (имена признаков)")
print("  - trained_models/model_info.pkl (информация о модели)")
print("  - *.png (графики)")

print("\nАнализ завершен!")