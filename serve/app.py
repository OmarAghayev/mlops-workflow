import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Bank Marketing Deposit Prediction API")

# Model ve bağımlı bileşenleri yükle
MODEL_PATH = "serve/model.joblib"

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    cat_cols = artifacts["cat_cols"]
except Exception as e:
    model, encoder, cat_cols = None, None, []
    print(f"Model yükleme uyarısı: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bank Marketing Prediction API is running."}

@app.post("/predict")
def predict(data: dict):
    if model is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model veya encoder henüz serve/model.joblib dizininde bulunamadı.")
    
    # 1. Gelen JSON verisini DataFrame'e çevir
    df = pd.DataFrame([data])
    
    # 2. Veri Sızıntısı Önleme
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
        
    # 3. Öznitelik Mühendisliği (Feature Engineering)
    df['previously_contacted'] = (df['pdays'] != 999).astype(int)
    df['pdays_cleaned'] = df['pdays'].apply(lambda x: 0 if x == 999 else x)
    df['total_contacts'] = df['campaign'] + df['previous']
    df['previous_success_ratio'] = df['poutcome'].apply(lambda x: 1 if x == 'success' else 0)
    
    # Zaman kaymasına yol açan makroekonomik değişkenleri kaldır
    macro_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    df = df.drop(columns=[c for c in macro_cols if c in df.columns], errors='ignore')
    
    # 4. Kategorik Kodlama (OrdinalEncoder)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    df[cat_cols] = encoder.transform(df[cat_cols])
    
    # 5. Olasılık ve Sınıf Tahmini
    prob = float(model.predict_proba(df)[:, 1][0])
    prediction = int(prob > 0.5)
    
    return {
        "prediction_probability": round(prob, 4),
        "deposit_subscribed": prediction
    }