import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score
import numpy as np

# ==============================================================================
# 1. إعدادات الرسوم البيانية
# ==============================================================================
# إعدادات لضمان ظهور الأرقام بشكل واضح
plt.style.use('ggplot')
sns.set_style('whitegrid')

# ==============================================================================
# 2. الرسم البياني الأول: متوسط السعر حسب المنطقة
# ==============================================================================
def plot_price_by_district():
    try:
        df = pd.read_csv('damasdata.csv')
    except FileNotFoundError:
        print("Error: 'data.csv' not found.")
        return

    # استخراج المنطقة
    df['District'] = df['Location'].apply(lambda x: x.split(' - ')[1])
    
    # حساب متوسط السعر لكل منطقة
    avg_price = df.groupby('District')['Price (SYP)'].mean().sort_values(ascending=False)
    
    # تحويل الأسعار إلى مليار ليرة سورية لتسهيل القراءة
    avg_price_billion = avg_price / 1_000_000_000

    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_price_billion.index, y=avg_price_billion.values, palette='viridis')
    
    plt.title('Average Property Price by District (Damascus)', fontsize=16)
    plt.xlabel('District', fontsize=12)
    plt.ylabel('Average Price (Billion SYP)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('avg_price_by_district.png')
    print("Chart 1: 'avg_price_by_district.png' saved successfully.")
    # plt.show() # يمكنك إلغاء التعليق على هذا السطر لرؤية الرسم البياني مباشرة

# ==============================================================================
# 3. الرسم البياني الثاني: مقارنة الأداء الفعلي بالتنبؤ (Scatter Plot)
# ==============================================================================
def plot_actual_vs_predicted():
    try:
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').squeeze()
        model = joblib.load('best_price_predictor.joblib')
    except FileNotFoundError:
        print("Error: Required files (X_test.csv, y_test.csv, best_price_predictor.joblib) not found. Please run steps 01 and 02.")
        return

    # التنبؤ على بيانات الاختبار
    y_pred = model.predict(X_test)
    
    # تحويل الأسعار إلى مليار ليرة سورية
    y_test_billion = y_test / 1_000_000_000
    y_pred_billion = y_pred / 1_000_000_000
    
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    
    # رسم النقاط الفعلية مقابل المتوقعة
    plt.scatter(y_test_billion, y_pred_billion, alpha=0.6, color='darkblue')
    
    # رسم خط مثالي (y=x)
    max_val = max(y_test_billion.max(), y_pred_billion.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction Line')
    
    plt.title(f'Actual vs. Predicted Prices (R² = {r2:.4f})', fontsize=16)
    plt.xlabel('Actual Price (Billion SYP)', fontsize=12)
    plt.ylabel('Predicted Price (Billion SYP)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Chart 2: 'actual_vs_predicted.png' saved successfully.")
    # plt.show() # يمكنك إلغاء التعليق على هذا السطر لرؤية الرسم البياني مباشرة

# ==============================================================================
# 4. تشغيل الدوال
# ==============================================================================
if __name__ == '__main__':
    plot_price_by_district()
    plot_actual_vs_predicted()
