import pandas as pd
import joblib
import numpy as np

# ==============================================================================
# 1. تحميل النموذج المدرب
# ==============================================================================
try:
    model = joblib.load('best_price_predictor.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'real_estate_price_predictor.joblib' not found. Please run '02_model_training.py' first.")
    exit()

# ==============================================================================
# 2. تعريف بيانات العقار الجديد
# ==============================================================================
# يجب أن تكون هذه البيانات بنفس تنسيق الميزات التي تم تدريب النموذج عليها
# (Area, Bedrooms, Bathrooms, Nearby Schools, Hospitals, Shops, District_dummies, PropType_dummies)

# مثال لعقار جديد:
new_property_data = {
    'Area (sqm)': 150,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Nearby Schools': 2,
    'Nearby Hospitals': 1,
    'Nearby Shops': 10,
    'District': 'Al-MALKI', # المنطقة
    'Property Type': 'Apartment' # نوع العقار
}

# ==============================================================================
# 3. تجهيز البيانات للتنبؤ
# ==============================================================================

# أ. إنشاء DataFrame من بيانات العقار الجديد
new_df = pd.DataFrame([new_property_data])

# ب. استيراد قائمة جميع الأعمدة (الميزات) التي تم تدريب النموذج عليها
# سنقوم بتحميل X_train للحصول على قائمة الأعمدة الكاملة
try:
    X_train_cols = pd.read_csv('X_train.csv').columns
except FileNotFoundError:
    print("Error: 'X_train.csv' not found. Cannot determine feature order.")
    exit()

# ج. تطبيق One-Hot Encoding على العقار الجديد
new_df = pd.get_dummies(new_df, columns=['District', 'Property Type'], drop_first=False)

# د. مواءمة الأعمدة (Aligning Columns)
# يجب أن يحتوي العقار الجديد على نفس الأعمدة وبنفس الترتيب الذي تم تدريب النموذج عليه
# سنقوم بإنشاء صف يحتوي على أصفار لجميع الأعمدة ثم نملأ القيم الصحيحة
final_features = pd.DataFrame(0, index=[0], columns=X_train_cols)

# نقل القيم من new_df إلى final_features
for col in new_df.columns:
    if col in final_features.columns:
        final_features[col] = new_df[col].iloc[0]

# ==============================================================================
# 4. التنبؤ بالسعر
# ==============================================================================
predicted_price = model.predict(final_features)[0]

# ==============================================================================
# 5. عرض النتيجة
# ==============================================================================
print("\n--- New Property Details ---")
print(f"Location: {new_property_data['District']}")
print(f"Type: {new_property_data['Property Type']}")
print(f"Area: {new_property_data['Area (sqm)']} sqm")
print(f"Bedrooms: {new_property_data['Bedrooms']}")

print("\n--- Predicted Price ---")
print(f"Predicted Price (SYP): {predicted_price:,.0f} L.S.")
print(f"Predicted Price (USD - approx): ${predicted_price / 15000:,.0f} (assuming 1 USD = 15,000 SYP)")
