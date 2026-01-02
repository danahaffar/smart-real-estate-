import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor # ستحتاجين لتثبيت مكتبة xgboost: pip install xgboost
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================================================================
# 1. تحميل البيانات المُجهزة
# ==============================================================================
try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
    print("Training and testing data loaded successfully.")
except FileNotFoundError:
    print("Error: One or more processed data files (X_train.csv, etc.) not found. Please run '01_data_preparation.py' first.")
    exit()

# ==============================================================================
# 2. تعريف النماذج
# ==============================================================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
}

results = []
best_r2 = -np.inf
best_model_name = ""
best_model = None

# ==============================================================================
# 3. تدريب وتقييم النماذج
# ==============================================================================
print("\nStarting training and evaluation of 3 models...")

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # تدريب النموذج
    model.fit(X_train, y_train)
    
    # التنبؤ على بيانات الاختبار
    y_pred = model.predict(X_test)
    
    # حساب مقاييس التقييم
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mean_price = y_test.mean()
    mae_percentage = (mae / mean_price) * 100
    
    # حفظ النتائج
    results.append({
        'Model': name,
        'R2 Score': r2,
        'MAE (SYP)': f"{mae:,.0f}",
        'MAE %': f"{mae_percentage:.2f}%"
    })
    
    # تحديد أفضل نموذج
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

# ==============================================================================
# 4. عرض النتائج وحفظ أفضل نموذج
# ==============================================================================
results_df = pd.DataFrame(results)
print("\n==================================================")
print("             Model Comparison Summary             ")
print("==================================================")
print(results_df.to_markdown(index=False))
print("==================================================")

# حفظ أفضل نموذج
joblib.dump(best_model, 'best_price_predictor.joblib')
print(f"\n✅ Best Model Selected: {best_model_name} with R2 Score: {best_r2:.4f}")
print("Best model saved successfully as 'best_price_predictor.joblib'.")

# حفظ أهمية الميزات لأفضل نموذج (إذا لم يكن Linear Regression)
if best_model_name != "Linear Regression":
    feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importance.to_csv('feature_importance_best_model.csv', header=True)
    print("Feature importance for the best model saved to 'feature_importance_best_model.csv'.")
