import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. تحميل النماذج والبيانات
# ==============================================================================
try:
    price_model = joblib.load('best_xgb_tuned_model.joblib')

    df_features = pd.read_csv('processed_features.csv')
    df_original = pd.read_csv('damasdata.csv')

    X_train_cols = df_features.columns
    scaler = joblib.load('feature_scaler.joblib')

    print("API: All models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"API Error: {e}")
    exit()

# ==============================================================================
# 2. Schemas
# ==============================================================================
class PropertyInput(BaseModel):
    area_sqm: float
    bedrooms: int
    bathrooms: int
    nearby_schools: int
    nearby_hospitals: int
    nearby_shops: int
    location: str
    property_type: str

class SearchRecommendationInput(BaseModel):
    area_sqm: float
    bedrooms: int
    bathrooms: int
    nearby_schools: int
    nearby_hospitals: int
    nearby_shops: int
    location: str
    property_type: str
    max_price: float | None = None
    min_price: float | None = None
    top_n: int = 10


# ==============================================================================
# 3. FastAPI
# ==============================================================================
app = FastAPI(
    title="Smart Real Estate AI API",
    version="2.0.0"
)

# ==============================================================================
# 4. تجهيز ميزات السعر
# ==============================================================================
def preprocess_price_input(data: PropertyInput):
    df = pd.DataFrame([{
        'Area (sqm)': data.area_sqm,
        'Bedrooms': data.bedrooms,
        'Bathrooms': data.bathrooms,
        'Nearby Schools': data.nearby_schools,
        'Nearby Hospitals': data.nearby_hospitals,
        'Nearby Shops': data.nearby_shops,
        'District': data.location,
        'Property Type': data.property_type
    }])

    df = pd.get_dummies(df, columns=['District', 'Property Type'], drop_first=True)

    final = pd.DataFrame(0, index=[0], columns=X_train_cols)
    for col in df.columns:
        if col in final.columns:
            final[col] = df[col].iloc[0]

    return final

# ==============================================================================
# 5. تجهيز ميزات البحث (Recommendation)
# ==============================================================================
def preprocess_search_input(data: SearchRecommendationInput):
    df = pd.DataFrame([{
        'Area (sqm)': data.area_sqm,
        'Bedrooms': data.bedrooms,
        'Bathrooms': data.bathrooms,
        'Nearby Schools': data.nearby_schools,
        'Nearby Hospitals': data.nearby_hospitals,
        'Nearby Shops': data.nearby_shops,
        'Location': data.location,
        'Property Type': data.property_type
    }])

    df = pd.get_dummies(df)

    aligned = pd.DataFrame(0, index=[0], columns=df_features.columns)
    for col in df.columns:
        if col in aligned.columns:
            aligned[col] = df[col].iloc[0]

    numerical_cols = [
        'Area (sqm)', 'Bedrooms', 'Bathrooms',
        'Nearby Schools', 'Nearby Hospitals', 'Nearby Shops'
    ]
    aligned[numerical_cols] = scaler.transform(aligned[numerical_cols])

    weights = {
        'Area (sqm)': 2.0,
        'Bedrooms': 1.5,
        'Bathrooms': 1.0,
        'Nearby Schools': 0.5,
        'Nearby Hospitals': 0.5,
        'Nearby Shops': 0.5
    }
    for f, w in weights.items():
        aligned[f] *= w

    return aligned.values

# ==============================================================================
# 6. Endpoint التنبؤ بالسعر
# ==============================================================================
@app.post("/predict_price")
def predict_price(data: PropertyInput):
    features = preprocess_price_input(data)
    prediction = price_model.predict(features)[0]

    return {
        "predicted_price_syp": round(prediction),
        "formatted": f"{round(prediction):,.0f} SYP"
    }

# ==============================================================================
# 7. Endpoint التوصية حسب البحث
# ==============================================================================
from sklearn.metrics.pairwise import cosine_similarity

@app.post("/recommend_by_search")
def recommend_by_search(data: SearchRecommendationInput):
    try:
        user_vector = preprocess_search_input(data)

        similarities = cosine_similarity(
            user_vector,
            df_features.values
        )[0]

        top_idx = np.argsort(similarities)[::-1][:data.top_n]

        results = df_original.iloc[top_idx][
            ['ID', 'Title', 'Location', 'Property Type', 'Price (SYP)']
        ].copy()

        results['Similarity Score'] = similarities[top_idx]

        return {
            "recommendations": results.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
