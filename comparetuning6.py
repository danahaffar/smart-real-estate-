import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np

# ==============================================================================
# 1. تحميل البيانات
# ==============================================================================
try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
except FileNotFoundError:
    print("Error: Data files not found. Please run '01_data_preparation.py'.")
    exit()

# ==============================================================================
# 2. أداء النموذج الأساسي (Baseline XGBoost) - قبل الضبط
# ==============================================================================
print("--- 2. Training Baseline XGBoost Model ---")
# استخدام المعايير الافتراضية (أو المعايير التي استخدمت في 02_model_training.py)
xgb_baseline = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_baseline.fit(X_train, y_train)
y_pred_baseline = xgb_baseline.predict(X_test)

r2_baseline = r2_score(y_test, y_pred_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
mae_perc_baseline = (mae_baseline / y_test.mean()) * 100

print(f"Baseline R2: {r2_baseline:.4f}, Baseline MAE: {mae_perc_baseline:.2f}%")

# ==============================================================================
# 3. أداء النموذج المُحسَّن (Tuned XGBoost) - بعد الضبط
# ==============================================================================
print("\n--- 3. Running Hyperparameter Tuning (Grid Search) ---")

# تعريف النموذج والمعايير (نفس المعايير في 05_hyperparameter_tuning.py)
xgb = XGBRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
}
# تقليل عدد الخيارات لتسريع التشغيل في هذا الكود
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=0, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_

y_pred_tuned = best_xgb_model.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mae_perc_tuned = (mae_tuned / y_test.mean()) * 100

print(f"Tuned R2: {r2_tuned:.4f}, Tuned MAE: {mae_perc_tuned:.2f}%")

# ==============================================================================
# 4. إنشاء الرسم البياني للمقارنة
# ==============================================================================
metrics = ['R-squared (R²)', 'MAE Percentage Error']
baseline_values = [r2_baseline, mae_perc_baseline]
tuned_values = [r2_tuned, mae_perc_tuned]

# إعداد البيانات للرسم البياني
data = {
    'Metric': metrics * 2,
    'Value': baseline_values + tuned_values,
    'Model': ['Baseline XGBoost'] * 2 + ['Tuned XGBoost'] * 2
}
df_plot = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', hue='Model', data=df_plot, palette=['#1f77b4', '#ff7f0e'])

# إضافة القيم على الأعمدة
for index, row in df_plot.iterrows():
    plt.text(index % 2 - 0.2, row['Value'] + (0.01 if index < 2 else 0.5), 
             f'{row["Value"]:.2f}{"%" if index % 2 != 0 else ""}', 
             color='black', ha="center", fontsize=10)

plt.title('XGBoost Model Performance: Baseline vs. Tuned', fontsize=16)
plt.ylabel('Value', fontsize=12)
plt.xlabel('Evaluation Metric', fontsize=12)
plt.legend(title='Model Version')
plt.tight_layout()
plt.savefig('xgb_tuning_comparison.png')
print("\n✅ Comparison chart saved as 'xgb_tuning_comparison.png'.")
# plt.show() # يمكنك إلغاء التعليق لرؤية الرسم البياني مباشرة
