import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# بيانات العقارات
properties = pd.DataFrame({
    'property_id': [1, 2, 3, 4],
    'city': ['Damascus', 'Damascus', 'Aleppo', 'Homs'],
    'price': [200000, 250000, 150000, 180000],
    'area': [120, 150, 100, 110],
    'rooms': [3, 4, 2, 3],
    'type': ['Apartment', 'Apartment', 'House', 'Apartment']
})

# بحث المستخدم
user_search = pd.DataFrame({
    'city': ['Damascus'],
    'price': [220000],
    'area': [130],
    'rooms': [3],
    'type': ['Apartment']
})

# دمج البيانات
combined = pd.concat(
    [user_search, properties.drop('property_id', axis=1)],
    ignore_index=True
)

# One-Hot Encoding (الإصدار الجديد)
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(combined[['city', 'type']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# الخصائص الرقمية
numerical = combined[['price', 'area', 'rooms']].reset_index(drop=True)
final_features = pd.concat([numerical, encoded_df], axis=1)

# حساب التشابه
similarity = cosine_similarity(final_features)
properties['similarity_score'] = similarity[0][1:]

# عرض النتائج
print(properties.sort_values(by='similarity_score', ascending=False))
