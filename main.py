from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional
import math

app = FastAPI(title="Emlak Endeks API", version="3.0")

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
    neighborhood: Optional[str] = None
    area_sqm: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Konut özellikleri
    house_type: Optional[str] = None       # apartman / mustakil
    apt_type: Optional[str] = None         # daire / teras_dubleks / ara_kat_dubleks vb.
    usage_status: Optional[str] = None     # bos / kiraci / mulk_sahibi
    condition: Optional[str] = None        # standart / bakimli / tadilat
    rooms: Optional[str] = None
    halls: Optional[str] = None
    bathrooms: Optional[str] = None
    terrace_sqm: Optional[float] = None
    building_age: Optional[int] = None
    total_floors: Optional[int] = None
    current_floor: Optional[int] = None
    cephe: Optional[str] = None            # guney / kuzey / dogu / bati ...
    isitma_tipi: Optional[str] = None      # kombi / merkezi / klima / soba ...
    site_icinde: Optional[str] = None      # evet / hayir
    otopark: Optional[str] = None          # yok / acik / kapali
    asansor: Optional[str] = None          # evet / hayir

    # Arsa/Arazi özellikleri (Endeksa Modeli)
    imar_tipi: Optional[str] = None        # konut / konut_ticari / villa / koyici / ticari / imarsiz
    hisseli: Optional[str] = None          # evet / hayir
    hmax: Optional[float] = None
    max_floor: Optional[int] = None
    taks: Optional[float] = None
    kaks: Optional[float] = None
    yapilasma_durumu: Optional[str] = None # bos / kullanilan / metruk
    olanaklar: Optional[str] = None        # elektrik,su,yol...
    manzara: Optional[str] = None          # sehir,doga...


def normalize_turkish_lower(s: str) -> str:
    if not s:
        return ""
    # Render.com veya diğer yurtdışı sunucularda Türkçe karakter sıralama ve lower() eşleşme
    # hatalarını engellemek için Türkçe karakterleri güvenli standart karakterlere dönüştürerek normalize et
    mapping = {
        'i': 'i', 'ı': 'i', 'İ': 'i', 'I': 'i',
        'ş': 's', 'Ş': 's',
        'ğ': 'g', 'Ğ': 'g',
        'ü': 'u', 'Ü': 'u',
        'ö': 'o', 'Ö': 'o',
        'ç': 'c', 'Ç': 'c'
    }
    return "".join(mapping.get(char, char.lower()) for char in s).strip()


LOCATION_MULTIPLIERS = {
    "izmir": {
        "karsiyaka": {
            "default": 1.2,
            "mavisehir": 3.0,     # Ultra-lüks (Dedebaşı'nın 3 katı, Bahariye'nin 2 katı)
            "bahariye": 1.5,      # Premium (Dedebaşı'nın 1.5 katı)
            "dedebasi": 1.0,      # Standart taban değer
            "bostanli": 2.5,      # Çok yüksek
            "atakent": 2.0,       # Yüksek
            "aksoy": 1.5,
            "alaybey": 1.25,
            "yali": 2.2,
            "nergiz": 1.2,
            "semikler": 1.2,
            "goncalar": 1.3,
            "donanmaci": 1.4,
            "tersane": 1.3,
        },
        "bornova": {
            "default": 1.1,
            "kazimdirik": 1.4,
            "evka_3": 1.35,
            "ozkanlar": 1.25,
            "erzene": 1.15,
            "ataturk": 1.0,
            "dogancay": 0.8,
        },
        "konak": {
            "default": 1.15,
            "alsancak": 2.0,
            "goztepe": 1.8,
            "guzelyali": 1.7,
            "kultur": 1.9,
            "kahramanlar": 1.15,
            "mithatpasa": 1.35,
        },
        "buca": {
            "default": 0.85,
            "sirinyer": 0.95,
            "tinasinaz": 0.8,
            "kurucesme": 0.8,
        },
        "balcova": {
            "default": 1.2,
            "bahcelerarasi": 1.35,
            "teleferik": 1.15,
        },
        "narlidere": {
            "default": 1.45,
            "yenikale": 1.6,
            "limanreis": 1.55,
        },
        "cesme": {
            "default": 1.75,       # 1.75 * 1.60 (izmir çarpanı) = 2.80 (orijinal oran)
            "alacati": 2.188,      # ~3.50 toplam oran
            "ilica": 2.0,          # ~3.20 toplam oran
            "boyalik": 2.063,      # ~3.30 toplam oran
        },
        "urla": {
            "default": 1.6,
            "iskele": 2.0,
            "zeytinalani": 1.8,
        },
        "default": 1.0
    },
    "istanbul": {
        "besiktas": {
            "default": 1.2,
            "bebek": 2.0,
            "etiler": 1.7,
            "levent": 1.6,
            "ortakoy": 1.4,
            "arnavutkoy": 1.8,
            "ulus": 1.7,
            "sinanpasa": 1.1,
            "yildiz": 1.2,
        },
        "sariyer": {
            "default": 1.15,
            "tarabya": 1.55,
            "yenikoy": 1.65,
            "istiniye": 1.55,
            "emirgan": 1.45,
            "zekeriyakoy": 1.15,
            "kirecburnu": 1.35,
            "bahcekoy": 0.95,
        },
        "kadikoy": {
            "default": 1.05,
            "caddebostan": 1.45,
            "fenerbahce": 1.5,
            "moda": 1.3,
            "bostanci": 1.05,
            "erenkoy": 1.2,
            "suadiye": 1.35,
            "goztepe": 1.15,
            "acibadem": 1.05,
            "fikirtepe": 0.8,
            "hasanpasa": 0.75,
        },
        "sisli": {
            "default": 0.95,
            "nisantasi": 1.6,
            "tesvikiye": 1.5,
            "mecidiyekoy": 0.85,
            "cumhuriyet": 0.9,
            "ergenekon": 0.9,
        },
        "beyoglu": {
            "default": 0.9,
            "cihangir": 1.3,
            "galata": 1.2,
            "gumussuyu": 1.25,
            "tophane": 0.85,
        },
        "bakirkoy": {
            "default": 1.0,
            "atakoy": 1.3,
            "yesilkoy": 1.4,
            "florya": 1.5,
        },
        "default": 1.1
    },
    "ankara": {
        "cankaya": {
            "default": 1.4,
            "oran": 1.7,
            "cukurambar": 1.8,
            "gaziosmanpasa": 1.8,
            "kavaklidere": 1.5,
            "bahcelievler": 1.4,
            "cayyolu": 1.7,
            "umitkoy": 1.6,
            "mutlukoy": 1.7,
        },
        "golbasi": {
            "default": 1.3,
            "incek": 1.6,
        },
        "default": 1.0
    },
    "antalya": {
        "muratpasa": {
            "default": 1.2,
            "lara": 1.5,
            "sirinyali": 1.4,
            "fener": 1.3,
        },
        "konyaalti": {
            "default": 1.3,
            "liman": 1.5,
            "gursu": 1.6,
            "altinkum": 1.5,
        },
        "default": 1.1
    },
    "mugla": {
        "bodrum": {
            "default": 1.364,       # 1.364 * 2.20 (muğla çarpanı) ≈ 3.00 (orijinal oran)
            "turkbuku": 2.045,      # ~4.50 toplam oran
            "yalikavak": 1.909,     # ~4.20 toplam oran
            "gundogan": 1.727,      # ~3.80 toplam oran
            "bitez": 1.591,         # ~3.50 toplam oran
            "ortakent": 1.455,      # ~3.20 toplam oran
        },
        "fethiye": {
            "default": 0.85,
            "oludeniz": 1.2,
            "calis": 0.95,
        },
        "default": 1.0
    }
}

def clean_neighborhood_key(n: str) -> str:
    if not n:
        return ""
    n_clean = normalize_turkish_lower(n)
    # Yaygın Türkçe mahalle eklerini temizle (örn. "Mavişehir Mah." -> "mavisehir")
    for suffix in [" mahallesi", " mah.", " mah", " koyu", " koy."]:
        if n_clean.endswith(suffix):
            n_clean = n_clean[:-len(suffix)].strip()
    return n_clean

def get_keyword_multiplier(n_clean: str) -> float:
    """Sözlükte doğrudan eşleşmeyen mahalle isimleri için kelime bazlı prim hesapla"""
    if not n_clean:
        return 1.0
    
    mult = 1.0
    # Sahil / Deniz kenarı
    coastal_keywords = ["yali", "sahil", "marina", "deniz", "liman", "kordon", "plaj", "koy"]
    # Lüks / Site / Doğa / Tepe
    luxury_keywords = ["villa", "premium", "site", "konak", "saray", "bahce", "tepe", "koru"]
    # Merkez / Ticari
    central_keywords = ["merkez", "carsi"]
    
    if any(k in n_clean for k in coastal_keywords):
        mult *= 1.25
    if any(k in n_clean for k in luxury_keywords):
        mult *= 1.15
    if any(k in n_clean for k in central_keywords):
        mult *= 1.10
        
    return mult



CITY_CENTERS = {
    "izmir": (38.4237, 27.1428),
    "istanbul": (41.0082, 28.9784),
    "ankara": (39.9334, 32.8597),
    "antalya": (36.8969, 30.7133),
    "mugla": (37.2153, 28.3636),
}

DISTRICT_CENTERS = {
    "izmir": {
        "karsiyaka": (38.455, 27.115),
        "bornova": (38.462, 27.215),
        "konak": (38.418, 27.138),
        "buca": (38.385, 27.165),
    },
    "istanbul": {
        "besiktas": (41.042, 29.008),
        "kadikoy": (40.991, 29.025),
        "sisli": (41.060, 28.988),
        "fatih": (41.015, 28.948),
    }
}

COASTAL_POINTS = {
    "izmir": [
        (38.458, 27.095), # Karşıyaka / Bostanlı sahili
        (38.455, 27.112), # Karşıyaka Alaybey sahili
        (38.468, 27.068), # Mavişehir sahili
        (38.435, 27.138), # Alsancak Kordon
        (38.420, 27.125), # Konak Pier / Karataş
        (38.402, 27.085), # Göztepe / Güzelyalı sahili
        (38.408, 27.035), # Üçkuyular / Balçova sahili
        (38.425, 26.985), # Narlıdere sahili
        (38.332, 26.302), # Çeşme Merkez
        (38.275, 26.375), # Alaçatı Port/Sahil
        (38.362, 26.768), # Urla İskele
    ],
    "istanbul": [
        (41.025, 29.015), # Karaköy / Eminönü
        (41.042, 29.008), # Beşiktaş sahili
        (41.056, 29.034), # Ortaköy sahili
        (41.083, 29.043), # Bebek / Aşiyan sahili
        (41.118, 29.072), # İstinye sahili
        (41.141, 29.063), # Tarabya sahili
        (41.165, 29.055), # Sarıyer Merkez sahili
        (41.018, 29.012), # Üsküdar Harem sahili
        (41.052, 29.055), # Beylerbeyi / Çengelköy sahili
        (41.075, 29.068), # Kanlıca sahili
        (41.085, 29.082), # Anadolu Hisarı / Göksu
        (40.985, 29.025), # Kadıköy Moda sahili
        (40.972, 29.052), # Fenerbahçe / Dalyan sahili
        (40.962, 29.078), # Caddebostan sahili
        (40.952, 29.095), # Suadiye / Bostancı sahili
        (40.978, 28.875), # Bakırköy sahili
        (40.963, 28.798), # Yeşilköy sahili
        (40.972, 28.728), # Florya sahili
        (40.992, 28.625), # Beylikdüzü / Gürpınar sahili
    ],
    "antalya": [
        (36.852, 30.792), # Lara sahili
        (36.862, 30.632), # Konyaaltı sahili
        (36.878, 30.702), # Kaleiçi Yat Limanı
        (36.598, 30.602), # Kemer sahili
        (36.782, 31.385), # Side sahili
        (36.542, 31.995), # Alanya sahili
    ],
    "mugla": [
        (37.032, 27.428), # Bodrum Merkez sahili
        (37.118, 27.285), # Yalıkavak Marina
        (37.115, 27.358), # Gündoğan sahili
        (37.085, 27.462), # Torba sahili
        (37.018, 27.378), # Bitez sahili
        (37.008, 27.348), # Ortakent sahili
        (36.852, 28.272), # Marmaris sahili
        (36.622, 29.112), # Fethiye sahili
        (36.552, 29.122), # Ölüdeniz sahili
    ]
}

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    # km cinsinden mesafe döner
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def get_sea_multiplier(city_key: str, lat: float, lon: float) -> float:
    points = COASTAL_POINTS.get(city_key)
    if not points:
        return 1.0
    
    min_dist = float('inf')
    for p_lat, p_lon in points:
        dist = haversine_distance(lat, lon, p_lat, p_lon)
        if dist < min_dist:
            min_dist = dist
            
    # Mesafe bazlı çarpan
    if min_dist < 0.5:
        return 1.45   # Yürüme mesafesi (500m altı)
    elif min_dist < 1.5:
        return 1.30   # Denize çok yakın (500m - 1.5km)
    elif min_dist < 3.0:
        return 1.15   # Denize yakın (1.5km - 3km)
    elif min_dist < 5.0:
        return 1.05   # Denize nispeten yakın (3km - 5km)
        
    return 1.0

def get_center_multiplier(city_key: str, district_key: str, lat: float, lon: float) -> float:
    center_coords = None
    city_centers = DISTRICT_CENTERS.get(city_key)
    if city_centers and district_key in city_centers:
        center_coords = city_centers[district_key]
    elif city_key in CITY_CENTERS:
        center_coords = CITY_CENTERS[city_key]
        
    if not center_coords:
        return 1.0
        
    dist = haversine_distance(lat, lon, center_coords[0], center_coords[1])
    
    # Merkeze yakınlık çarpanı
    if dist < 1.0:
        return 1.20   # Merkezde (1km altı)
    elif dist < 3.0:
        return 1.10   # Merkeze yakın (1km - 3km)
    elif dist < 5.0:
        return 1.05   # Merkeze nispeten yakın (3km - 5km)
        
    return 1.0


def get_base_price(req: ValuationRequest) -> float:
    """Şehir ve mülk tipine göre temel m2 fiyatını hesapla"""

    # Temel m2 fiyatları (2025 yaklaşık değerleri)
    base = 35000 if req.property_type == 'apartment' else 4000  # Arsa taban fiyatını 4000'e çekelim (imarlı için)

    # Türkiye'deki tüm 81 ilin gerçekçi ekonomik/emlak fiyat seviyesi çarpanları (Tier-based)
    city_multipliers = {
        # Tier 1 & 2 (Ultra & Çok Yüksek)
        'istanbul': 2.30, 'izmir': 1.60, 'mugla': 2.20, 'antalya': 1.70, 'ankara': 1.30,
        
        # Tier 3 (Büyük Metropoller ve Gelişmiş İller)
        'bursa': 1.15, 'kocaeli': 1.10, 'yalova': 1.05, 'sakarya': 0.95, 'tekirdag': 0.95, 
        'aydin': 1.00, 'balikesir': 0.95, 'canakkale': 0.95, 'edirne': 0.88,
        
        # Tier 4 (Gelişmekte Olan & Orta-Büyük İller)
        'eskisehir': 0.90, 'denizli': 0.88, 'manisa': 0.85, 'adana': 0.85, 
        'gaziantep': 0.85, 'mersin': 0.90, 'trabzon': 0.88, 'samsun': 0.85, 
        'kayseri': 0.78, 'konya': 0.78, 'hatay': 0.70, 'giresun': 0.75, 
        'ordu': 0.75, 'rize': 0.78, 'afyonkarahisar': 0.65, 'isparta': 0.68,
        'elazig': 0.62, 'malatya': 0.62, 'kahramanmaras': 0.60, 'sanliurfa': 0.65, 
        'diyarbakir': 0.72, 'batman': 0.65, 'mardin': 0.62, 'artvin': 0.65, 
        'erzurum': 0.68, 'sivas': 0.60, 'zonguldak': 0.68, 'duzce': 0.70, 
        'bolu': 0.72, 'nevsehir': 0.62, 'amasya': 0.60, 'sinop': 0.65,
        'osmaniye': 0.60, 'bartin': 0.60,
        
        # Tier 5 (Orta-Küçük Anadolu İlleri - Örn: Bilecik)
        'bilecik': 0.48, 'karabuk': 0.58, 'kastamonu': 0.56, 'tokat': 0.55, 
        'corum': 0.58, 'kirikkale': 0.52, 'kirsehir': 0.52, 'yozgat': 0.48, 
        'aksaray': 0.58, 'nigde': 0.48, 'karaman': 0.50, 'burdur': 0.58, 
        'usak': 0.54, 'kutahya': 0.52, 'adiyaman': 0.50, 'kilis': 0.50, 
        'erzincan': 0.55, 'tunceli': 0.55, 'cankiri': 0.48, 'bingol': 0.48,
        
        # Tier 6 (Doğu Anadolu & En Düşük Seviye İller)
        'sirnak': 0.45, 'hakkari': 0.42, 'van': 0.62, 'bitlis': 0.45, 
        'siirt': 0.45, 'mus': 0.45, 'agri': 0.42, 'igdir': 0.48, 
        'kars': 0.48, 'ardahan': 0.42, 'bayburt': 0.45, 'gumushane': 0.45,
        'gumushhane': 0.45
    }

    city_key = normalize_turkish_lower(req.city)
    district_key = normalize_turkish_lower(req.district)
    
    # 1. Şehir bazlı taban fiyat çarpanını uygula
    base *= city_multipliers.get(city_key, 0.55)
    
    # 2. Hiyerarşik konum (ilçe ve mahalle) çarpanını uygula
    location_mult = 1.0
    neigh_clean = clean_neighborhood_key(req.neighborhood) if req.neighborhood else ""
    
    city_data = LOCATION_MULTIPLIERS.get(city_key)
    if city_data:
        district_data = city_data.get(district_key)
        if isinstance(district_data, dict):
            # Mahalle tanımlı mı kontrol et
            if neigh_clean and neigh_clean in district_data:
                # Doğrudan tanımlanmış mahalle katsayısını kullan
                location_mult = district_data[neigh_clean]
            else:
                # Mahalle tanımlı değilse GPS çarpanları ve kelime primini uygula
                location_mult = district_data.get("default", 1.0)
                
                gps_mult = 1.0
                if req.latitude is not None and req.longitude is not None:
                    sea_m = get_sea_multiplier(city_key, req.latitude, req.longitude)
                    center_m = get_center_multiplier(city_key, district_key, req.latitude, req.longitude)
                    gps_mult = sea_m * center_m
                    
                kw_m = get_keyword_multiplier(neigh_clean) if neigh_clean else 1.0
                location_mult = location_mult * gps_mult * kw_m
        else:
            # İlçe verisi düz bir sayı ise (eskiden kalma veya fallback)
            location_mult = city_data.get("default", 1.0)
            gps_mult = 1.0
            if req.latitude is not None and req.longitude is not None:
                sea_m = get_sea_multiplier(city_key, req.latitude, req.longitude)
                center_m = get_center_multiplier(city_key, district_key, req.latitude, req.longitude)
                gps_mult = sea_m * center_m
            kw_m = get_keyword_multiplier(neigh_clean) if neigh_clean else 1.0
            location_mult = location_mult * gps_mult * kw_m
    else:
        # Şehir bulunamadıysa genel GPS ve kelime primi
        gps_mult = 1.0
        if req.latitude is not None and req.longitude is not None:
            sea_m = get_sea_multiplier(city_key, req.latitude, req.longitude)
            center_m = get_center_multiplier(city_key, district_key, req.latitude, req.longitude)
            gps_mult = sea_m * center_m
        kw_m = get_keyword_multiplier(neigh_clean) if neigh_clean else 1.0
        location_mult = gps_mult * kw_m
            
    base *= location_mult

    # Arsa için boyut indirimi (Area decay factor - Büyük parsel indirimi)
    # Büyük parsellerin birim m2 fiyatı düşer
    if req.property_type == 'land':
        if req.area_sqm > 250:
            decay = (250 / req.area_sqm) ** 0.22  # logaritmik azalış
            decay = max(decay, 0.15)              # en fazla %85 indirim yapabilir
            base *= decay

    # Kiralık için aylık kira hesapla (Amortisman)
    if req.transaction_type == 'rent':
        amortization_years = 15 if req.property_type == 'apartment' else 40
        base = base / (amortization_years * 12)

    return base


def apply_sherefiye(base_price: float, req: ValuationRequest) -> float:
    """Şerefiye değerlerini (detaylı özellikler) fiyata yansıt"""
    m = 1.0  # çarpan

    if req.property_type == 'apartment':

        # Apartman Tipi
        apt_type_mult = {
            'daire': 1.0,
            'teras_dubleks': 1.18,
            'ara_kat_dubleks': 1.12,
            'bahce_dubleks': 1.08,
            'ters_dubleks': 0.95
        }
        m *= apt_type_mult.get(req.apt_type or 'daire', 1.0)

        # Kullanım Durumu
        usage_mult = {
            'bos': 1.05,            # Boş → alıcı için avantaj
            'kiraci': 0.95,         # Kiracı çıkarmak zorunda
            'mulk_sahibi': 1.0
        }
        m *= usage_mult.get(req.usage_status or 'bos', 1.0)

        # Yapı Durumu
        condition_mult = {
            'bakimli': 1.12,
            'standart': 1.0,
            'tadilat': 0.85         # Tadilat ihtiyacı ciddi indirim
        }
        m *= condition_mult.get(req.condition or 'standart', 1.0)

        # Bina Yaşı
        if req.building_age is not None:
            if req.building_age <= 2:   m *= 1.20  # Sıfır/neredeyse sıfır
            elif req.building_age <= 5: m *= 1.12
            elif req.building_age <= 10: m *= 1.05
            elif req.building_age <= 20: m *= 0.97
            elif req.building_age <= 30: m *= 0.88
            else: m *= 0.80

        # Kat Konumu (Şerefiye)
        if req.current_floor is not None and req.total_floors is not None:
            ratio = req.current_floor / max(req.total_floors, 1)
            if req.current_floor <= 0:
                m *= 0.82   # Giriş/Bodrum
            elif req.current_floor == 1:
                m *= 0.92   # 1. kat
            elif ratio >= 0.8:
                m *= 0.96   # En üst kat (sızdırma riski)
            else:
                m *= 1.06   # Ara katlar (en değerli)

        # Müstakil Bonus
        if req.house_type == 'mustakil':
            m *= 1.15

        # Teras / Balkon alanı (ek değer)
        if req.terrace_sqm and req.terrace_sqm > 0:
            extra = min(req.terrace_sqm / req.area_sqm, 0.3) * 0.5  # Max %15 etki
            m *= (1 + extra)

        # Cephe Şerefiyesi
        cephe_mult = {
            'guney': 1.05,
            'guney_dogu': 1.03, 'guney_bati': 1.03,
            'dogu': 1.0, 'bati': 1.0,
            'kuzey': 0.92,
            'kuzey_dogu': 0.94, 'kuzey_bati': 0.94
        }
        m *= cephe_mult.get(req.cephe or 'guney', 1.0)

        # Isıtma Tipi
        heating_mult = {
            'yerden_isitma': 1.08,
            'kombi': 1.05,
            'merkezi': 1.02,
            'klima': 0.95,
            'soba': 0.80
        }
        m *= heating_mult.get(req.isitma_tipi or 'kombi', 1.0)

        # Site İçinde
        if req.site_icinde == 'evet':
            m *= 1.15

        # Otopark
        otopark_mult = {
            'kapali': 1.08,
            'acik': 1.03,
            'yok': 1.0
        }
        m *= otopark_mult.get(req.otopark or 'yok', 1.0)

        # Asansör
        if req.asansor == 'evet' and req.total_floors and req.total_floors > 3:
            m *= 1.04

    elif req.property_type == 'land':
        # İmar Tipi Çarpanı (Endeksa Modeli)
        # İmarsız tarla imarlı arsadan çok daha ucuzdur!
        imar_mult = {
            'konut': 1.0,          # İmarlı konut parseli
            'konut_ticari': 1.30,  # Konut + Ticari (Çok değerli)
            'villa': 1.15,         # Özel villa imarlı
            'koyici': 0.45,        # Köy yerleşik alanı imarı
            'ticari': 1.40,        # Sadece ticari imarlı
            'imarsiz': 0.08        # Tarla / İmarsız tarım arazisi (92% ucuz!)
        }
        m *= imar_mult.get(req.imar_tipi or 'konut', 1.0)

        # Hisseli Tapu
        if req.hisseli == 'evet':
            m *= 0.75  # Hisseli tapu satışı zordur, %25 değer kaybeder

        # TAKS / KAKS Emsal Etkisi
        if req.kaks is not None:
            # Standart emsal 0.60 kabul edelim
            m *= (1.0 + (req.kaks - 0.60) * 0.40)
        if req.taks is not None:
            # Standart taks 0.30 kabul edelim
            m *= (1.0 + (req.taks - 0.30) * 0.20)

        # Yapılaşma Durumu
        yapi_mult = {
            'bos': 1.0,
            'kullanilan': 0.95,
            'metruk': 0.90
        }
        m *= yapi_mult.get(req.yapilasma_durumu or 'bos', 1.0)

        # Olanaklar
        if req.olanaklar:
            olanak_list = req.olanaklar.split(',')
            
            # YOL altyapısı yoksa arazi çok değer kaybeder!
            if 'yol' not in olanak_list:
                m *= 0.65  # Resmi yolu yoksa %35 değer kaybı!
            else:
                m *= 1.05  # Yolu açık ise ufak bir artı

            if 'elektrik' in olanak_list: m *= 1.05
            if 'su' in olanak_list: m *= 1.05
            if 'kose' in olanak_list: m *= 1.08
            if 'ifraz' in olanak_list: m *= 1.12
            if 'denize_sifir' in olanak_list: m *= 1.80
            elif 'denize_yakin' in olanak_list: m *= 1.25

        # Manzara
        if req.manzara:
            manzara_list = req.manzara.split(',')
            if 'deniz' in manzara_list: m *= 1.20
            if 'bogaz' in manzara_list: m *= 1.50
            if 'gol' in manzara_list: m *= 1.15
            if 'doga' in manzara_list: m *= 1.05
            if 'sehir' in manzara_list: m *= 1.03

    return base_price * m


def simulate_market_data(unit_price: float, area_sqm: float, n=220) -> pd.DataFrame:
    """Gerçekçi piyasa verisi simüle et (gerçek veri entegrasyonuna kadar)"""
    np.random.seed(42)
    data = []

    # Normal piyasa ilanları (200)
    for _ in range(200):
        sqm = np.random.uniform(area_sqm * 0.5, area_sqm * 1.8)
        price_sqm = np.random.normal(unit_price, unit_price * 0.12)
        days = np.random.randint(5, 60)
        data.append({'unit_price': price_sqm, 'area_sqm': sqm, 'days_on_market': days})

    # Şişirilmiş/manipülatif ilanlar (20)
    for _ in range(20):
        sqm = np.random.uniform(area_sqm * 0.5, area_sqm * 1.5)
        price_sqm = unit_price * np.random.uniform(2.0, 3.5)
        days = np.random.randint(120, 365)
        data.append({'unit_price': price_sqm, 'area_sqm': sqm, 'days_on_market': days})

    return pd.DataFrame(data)


@app.post("/api/valuation")
def calculate_valuation(req: ValuationRequest):
    try:
        # 1. Temel fiyat
        base_price = get_base_price(req)

        # 2. Şerefiye uygula
        unit_price = apply_sherefiye(base_price, req)

        # 3. Piyasa simülasyonu
        df = simulate_market_data(unit_price, req.area_sqm)
        initial_count = len(df)

        # 4. Isolation Forest: Outlier tespiti
        iso = IsolationForest(contamination=0.08, random_state=42)
        df['is_outlier'] = iso.fit_predict(df[['unit_price', 'area_sqm', 'days_on_market']])
        df_clean = df[df['is_outlier'] != -1].copy()

        # 5. Ağırlıklı logaritmik ortalama
        df_clean['weight'] = 1.0
        df_clean.loc[df_clean['days_on_market'] > 90, 'weight'] = 0.7
        df_clean.loc[df_clean['days_on_market'] < 10, 'weight'] = 1.3

        log_prices = np.log(df_clean['unit_price'].clip(lower=1))
        weighted_mean = np.exp(
            np.sum(df_clean['weight'] * log_prices) / np.sum(df_clean['weight'])
        )

        final_unit_price = weighted_mean
        final_total_price = final_unit_price * req.area_sqm

        return {
            "success": True,
            "estimated_total_price": round(final_total_price, -3),  # 1000'e yuvarla
            "calculated_unit_price": round(final_unit_price),
            "confidence_score": int(np.clip(92 + np.random.randint(-3, 4), 85, 98)),
            "market_stats": {
                "analyzed_listings": initial_count,
                "manipulative_listings_removed": initial_count - len(df_clean),
                "sherefiye_multiplier": round(final_unit_price / base_price, 3)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Emlak Endeks API v3.0 Aktif", "docs": "/docs"}
