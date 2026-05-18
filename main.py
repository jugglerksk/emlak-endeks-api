from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Emlak Endeks API - Detaylı", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ValuationRequest(BaseModel):
    transaction_type: str 
    property_type: str    
    city: str
    district: str
    neighborhood: str = None # Yeni eklendi: Mahalle
    area_sqm: float
    rooms: str = None
    building_age: int = None # Yeni: Bina Yaşı
    total_floors: int = None # Yeni: Kat Sayısı
    current_floor: int = None # Yeni: Bulunduğu Kat
    zoning: str = None

def get_historical_data(req: ValuationRequest) -> pd.DataFrame:
    np.random.seed(42)
    
    # 1. ŞEHİR BAZLI TEMEL FİYAT HESAPLAMASI
    # Türkiye Geneli Ortalama (M2)
    base_price = 25000 if req.property_type == 'apartment' else 6000
    
    # Şehir Çarpanları
    city = req.city.lower()
    if city == 'istanbul': base_price *= 2.2
    elif city == 'izmir': base_price *= 1.4
    elif city == 'ankara': base_price *= 1.2
    elif city == 'antalya': base_price *= 1.6
    elif city == 'mugla': base_price *= 2.0
    
    # Kiralık ise satışı kiraya çevir (Amortisman)
    if req.transaction_type == 'rent':
        base_price = base_price / (18 * 12) if req.property_type == 'apartment' else base_price / (40 * 12)

    # 2. ŞEREFİYE VE DETAYLI ÖZELLİK ÇARPANLARI
    multiplier = 1.0
    
    if req.property_type == 'apartment':
        # Yeni binalar daha değerli
        if req.building_age is not None:
            if req.building_age <= 5: multiplier *= 1.15
            elif req.building_age > 20: multiplier *= 0.85
            
        # Ara katlar daha değerli, bodrum/giriş daha ucuz
        if req.current_floor is not None and req.total_floors is not None:
            if req.current_floor <= 0: multiplier *= 0.80 # Giriş veya bodrum
            elif req.current_floor == req.total_floors: multiplier *= 0.95 # En üst kat
            else: multiplier *= 1.05 # Ara kat
            
    elif req.property_type == 'land' and req.zoning == 'imarli':
        multiplier *= 1.5

    base_price *= multiplier

    # Veri Üretimi
    data = []
    for _ in range(200):
        sqm = np.random.uniform(80, 200) if req.property_type == 'apartment' else np.random.uniform(300, 1000)
        price_sqm = np.random.normal(base_price, base_price * 0.1)
        data.append({'price': price_sqm * sqm, 'area_sqm': sqm, 'time_on_market': np.random.randint(5, 45)})
        
    for _ in range(10): # Outlier (Sahte) Veriler
        sqm = np.random.uniform(80, 200) if req.property_type == 'apartment' else np.random.uniform(300, 1000)
        price_sqm = base_price * np.random.uniform(2.5, 4.0) 
        data.append({'price': price_sqm * sqm, 'area_sqm': sqm, 'time_on_market': np.random.randint(90, 180)})
        
    return pd.DataFrame(data)

@app.post("/api/valuation")
def calculate_valuation(req: ValuationRequest):
    try:
        df = get_historical_data(req)
        df['unit_price'] = df['price'] / df['area_sqm']
        initial_count = len(df)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['is_outlier'] = iso_forest.fit_predict(df[['unit_price', 'area_sqm']])
        df_clean = df[df['is_outlier'] != -1].copy()
        
        df_clean['weight'] = 1.0
        df_clean.loc[df_clean['time_on_market'] > 90, 'weight'] = 0.8
        df_clean.loc[df_clean['time_on_market'] < 15, 'weight'] = 1.2
        
        weighted_log_sum = np.sum(df_clean['weight'] * np.log(df_clean['unit_price']))
        total_weight = np.sum(df_clean['weight'])
        
        calculated_unit_price = np.exp(weighted_log_sum / total_weight)
        estimated_total_price = calculated_unit_price * req.area_sqm
        
        return {
            "success": True,
            "estimated_total_price": round(estimated_total_price),
            "calculated_unit_price": round(calculated_unit_price),
            "confidence_score": 92 + np.random.randint(-2, 3),
            "market_stats": {
                "analyzed_listings": initial_count,
                "manipulative_listings_removed": initial_count - len(df_clean)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Emlak Endeks API V2 Çalışıyor."}
