import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. تحميل البيانات
# ==============================================================================
try:
    df = pd.read_csv('damasdata.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'damascus_real_estate_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# ==============================================================================
# 2. تنظيف البيانات (Data Cleaning)
# ==============================================================================
# التأكد من القيود
df = df[(df['Bedrooms'] > 0) & (df['Bathrooms'] > 0)]
print("\nConstraints confirmed: Bedrooms and Bathrooms are all > 0.")

# ==============================================================================
# 3. هندسة الميزات (Feature Engineering)
# ==============================================================================

# أ. استخراج المنطقة (District) من عمود الموقع (Location)
df['District'] = df['Location'].apply(lambda x: x.split(' - ')[1])

# ب. ترميز المتغيرات الفئوية (Categorical Encoding)
# استخدام One-Hot Encoding للمنطقة ونوع العقار
df = pd.get_dummies(df, columns=['District', 'Property Type'], drop_first=True)

# ج. حذف الأعمدة النصية الأصلية والأعمدة غير الضرورية
columns_to_drop = ['ID', 'Title', 'Description', 'Listing Date', 'Location']
df.drop(columns=columns_to_drop, inplace=True)

# ==============================================================================
# 4. تقسيم البيانات إلى ميزات (X) ومتغير مستهدف (y)
# ==============================================================================
# المتغير المستهدف هو السعر (Price)
X = df.drop(columns=['Price (SYP)'])
y = df['Price (SYP)']

# ==============================================================================
# 5. حفظ البيانات المعالجة (X) في ملف processed_features.csv
# ==============================================================================
# هذا الملف يحتوي على جميع الميزات الرقمية والترميزية التي يحتاجها نظام التوصية
X.to_csv('processed_features.csv', index=False)

# ==============================================================================
# 6. تقسيم البيانات إلى تدريب واختبار (Train/Test Split)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# حفظ البيانات المقسمة لتدريب النموذج
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\n--- Data Preparation Summary ---")
print(f"Final features shape (X): {X.shape}")
print(f"Training set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print("Processed data saved to processed_features.csv and split files.")
print("You can now proceed to the next step: Model Training.")
