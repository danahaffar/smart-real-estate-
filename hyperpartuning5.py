import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ==============================================================================
# 1. تحميل البيانات المُجهزة
# ==============================================================================
try:
    # نحتاج إلى بيانات التدريب الكاملة
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    print("Training data loaded successfully.")
except FileNotFoundError:
    print("Error: 'X_train.csv' or 'y_train.csv' not found. Please run '01_data_preparation.py' first.")
    exit()

# ==============================================================================
# 2. تعريف النموذج والمعايير المراد اختبارها
# ==============================================================================
# تعريف نموذج XGBoost
xgb = XGBRegressor(random_state=42, n_jobs=-1)

# تعريف شبكة المعايير (Hyperparameter Grid)
# سنقوم باختبار مجموعة من المعايير الأكثر تأثيراً
param_grid = {
    'n_estimators': [100, 200, 300], # عدد الأشجار
    'max_depth': [3, 5, 7],          # أقصى عمق للشجرة
    'learning_rate': [0.05, 0.1, 0.2], # معدل التعلم
    'min_child_weight': [1, 3]       # الحد الأدنى لوزن العينة الفرعية
}

# ==============================================================================
# 3. إعداد Grid Search
# ==============================================================================
# استخدام GridSearchCV للبحث عن أفضل تركيبة معايير
# cv=5 تعني استخدام 5-fold Cross-Validation
# scoring='neg_mean_absolute_error' يعني أننا نسعى لتقليل الخطأ المطلق
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error', # الهدف هو تقليل الخطأ المطلق
    cv=5,
    verbose=2, # لعرض التقدم أثناء البحث
    n_jobs=-1
)

# ==============================================================================
# 4. تشغيل البحث عن أفضل المعايير
# ==============================================================================
print("\nStarting Hyperparameter Tuning (Grid Search)... This may take a few minutes.")
grid_search.fit(X_train, y_train)

# ==============================================================================
# 5. عرض النتائج وحفظ أفضل نموذج
# ==============================================================================
best_xgb_model = grid_search.best_estimator_

# حفظ أفضل نموذج
joblib.dump(best_xgb_model, 'best_xgb_tuned_model.joblib')

print("\n==================================================")
print("         Hyperparameter Tuning Results            ")
print("==================================================")
print(f"✅ Best Parameters Found: {grid_search.best_params_}")
# يتم تحويل النتيجة من سالب إلى موجب
best_mae = -grid_search.best_score_
print(f"✅ Best Cross-Validation MAE: {best_mae:,.0f} SYP")
print("==================================================")

# ==============================================================================
# 6. تقييم النموذج المُحسَّن على بيانات الاختبار
# ==============================================================================
try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()
except FileNotFoundError:
    print("Error: 'X_test.csv' or 'y_test.csv' not found.")
    exit()

y_pred_tuned = best_xgb_model.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mae_percentage_tuned = (mae_tuned / y_test.mean()) * 100

print("\n--- Evaluation of Tuned XGBoost Model on Test Set ---")
print(f"R-squared (R2) Score: {r2_tuned:.4f}")
print(f"Mean Absolute Error (MAE): {mae_tuned:,.0f} SYP")
print(f"MAE as Percentage of Mean Price: {mae_percentage_tuned:.2f}%")
print("\nTuned XGBoost model saved as 'best_xgb_tuned_model.joblib'.")
