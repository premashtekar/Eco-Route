"""
EcoRoute — Carbon-Aware Cyber-Secure Route Optimization (Indian cities demo)
- Uses ML models (LightGBM, RandomForest, IsolationForest)
- Integrates two datasets (carbon/traffic-like + optional VANET)
- Produces eco-secure routes and visualizes them on a Folium map
- If dataset lacks source/destination, edges are assigned to Indian city pairs
  (so map visualizations and route names use real Indian city names & coordinates)
- Animated vehicle-dominant edges using AntPath

Usage:
    pip install streamlit pandas numpy scikit-learn lightgbm joblib networkx folium streamlit-folium
    streamlit run main.py
"""
import os
import json
import math
import time
import hashlib
import hmac
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# map
import folium
from folium.plugins import AntPath

try:
    import importlib
    _sf = importlib.import_module("streamlit_folium")
    st_folium = getattr(_sf, "st_folium", None)
    if st_folium is None:
        raise ImportError("st_folium not found in streamlit_folium")
except Exception:
    # Fallback: render folium map HTML via streamlit components if streamlit_folium isn't installed.
    import streamlit.components.v1 as components
    def st_folium(m, width=700, height=500):
        try:
            if hasattr(m, "_repr_html_"):
                html = m._repr_html_()
            else:
                html = m.get_root().render()
        except Exception:
            html = "<div>Map could not be rendered</div>"
        components.html(html, width=width, height=height)

# -----------------------
# Config & paths
# -----------------------
st.set_page_config(page_title="EcoRoute — Carbon-Aware Cyber-Secure Route Optimization", layout="wide")
BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models_output"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CSV_PRIMARY = "carbon_aware_cybersecurity_dataset.csv"
CSV_VANET = "CCR_5G_VANETs.csv"
META_PATH = MODELS_DIR / "meta.json"
ADAPTIVE_PATH = MODELS_DIR / "adaptive_weights.json"
HMAC_FALLBACK = MODELS_DIR / "hmac_fallback.key"
HMAC_ENV = "ECOROUTE_HMAC_KEY"

DEFAULT_WEIGHTS = {'co2_weight': 0.4, 'traffic_weight': 0.4, 'security_weight': 0.2}
RANDOM_SEED = 42

# -----------------------
# Indian cities table (name -> lat, lon)
# -----------------------
INDIAN_CITIES = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Surat": (21.1702, 72.8311),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126)
}
CITY_NAMES = list(INDIAN_CITIES.keys())

# -----------------------
# Utilities: HMAC + adaptive
# -----------------------
def _get_hmac_key():
    key = os.environ.get(HMAC_ENV)
    if key:
        return key.encode('utf-8')
    if HMAC_FALLBACK.exists():
        return HMAC_FALLBACK.read_bytes()
    else:
        k = hashlib.sha256(str(datetime.datetime.now().timestamp()).encode()).digest()
        HMAC_FALLBACK.write_bytes(k)
        return k

def sign_dict(obj: Dict) -> str:
    key = _get_hmac_key()
    payload = json.dumps(obj, sort_keys=True, separators=(',',':')).encode('utf-8')
    return hmac.new(key, payload, hashlib.sha256).hexdigest()

def verify_signature(obj: Dict, sig: str) -> bool:
    return hmac.compare_digest(sign_dict(obj), sig)

def load_adaptive_weights() -> Dict[str,float]:
    if ADAPTIVE_PATH.exists():
        try:
            return json.loads(ADAPTIVE_PATH.read_text())
        except:
            pass
    ADAPTIVE_PATH.write_text(json.dumps(DEFAULT_WEIGHTS))
    return DEFAULT_WEIGHTS.copy()

def save_adaptive_weights(w: Dict[str,float]):
    ADAPTIVE_PATH.write_text(json.dumps(w))

def adapt_weights_on_feedback(prev_weights: Dict[str,float], feedback: Dict[str,float]) -> Dict[str,float]:
    w = prev_weights.copy()
    lr = 0.05
    if 'co2_error_ratio' in feedback:
        delta = min(0.2, max(-0.2, (feedback['co2_error_ratio'] - 1.0))) * lr
        w['co2_weight'] = max(0.0, min(1.0, w['co2_weight'] + delta))
    if 'time_error_ratio' in feedback:
        delta = min(0.2, max(-0.2, (feedback['time_error_ratio'] - 1.0))) * lr
        w['traffic_weight'] = max(0.0, min(1.0, w['traffic_weight'] + delta))
    if 'security_violation' in feedback:
        delta = lr * (1.0 if feedback['security_violation'] else -0.5)
        w['security_weight'] = max(0.0, min(1.0, w['security_weight'] + delta))
    s = sum(w.values()) or 1.0
    for k in w:
        w[k] = w[k] / s
    save_adaptive_weights(w)
    return w

# -----------------------
# Data loading & merge helpers
# -----------------------
def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            # Check if timestamp column exists before parsing
            df_sample = pd.read_csv(path, nrows=1)
            if 'timestamp' in df_sample.columns:
                return pd.read_csv(path, parse_dates=['timestamp'])
            else:
                return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path)
    return None

def merge_primary_and_vanet(primary_df: pd.DataFrame, vanet_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = primary_df.copy()
    if vanet_df is None:
        return df
    # time-bucket join: floor timestamps to 1 minute, group vanet by edge_id and minute
    vanet = vanet_df.copy()
    if 'timestamp' in vanet.columns:
        vanet['ts_min'] = pd.to_datetime(vanet['timestamp']).dt.floor('1min')
    else:
        vanet['ts_min'] = pd.Timestamp.now().floor('min')
    # ensure edge_id exists in vanet if possible
    if 'edge_id' not in vanet.columns:
        if {'source','destination'}.issubset(vanet.columns):
            vanet['edge_id'] = vanet['source'].astype(str) + "-" + vanet['destination'].astype(str)
        else:
            vanet['edge_id'] = 'unknown'
    agg_cols = [c for c in ['packet_loss','latency','signal_strength','gps_jump_flag'] if c in vanet.columns]
    if not agg_cols:
        return df
    agg = vanet.groupby(['edge_id','ts_min'])[agg_cols].agg(['mean']).reset_index()
    agg.columns = ['edge_id','timestamp'] + [f"{c}_mean" for c in agg_cols]
    # round primary timestamps into minutes
    if 'timestamp' in df.columns:
        df['ts_min'] = pd.to_datetime(df['timestamp']).dt.floor('1min')
    else:
        df['ts_min'] = pd.Timestamp.now().floor('min')
    merged = df.merge(agg, left_on=['edge_id','ts_min'], right_on=['edge_id','timestamp'], how='left', suffixes=('','_vanet'))
    for c in agg_cols:
        col_name = f"{c}_mean"
        if col_name in merged.columns:
            merged[col_name] = merged[col_name].fillna(0.0)
    merged = merged.drop(columns=['ts_min','timestamp_vanet'], errors='ignore')
    return merged

# -----------------------
# Feature engineering
# -----------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure source/destination exist; if not, create synthetic using city list
    if 'source' not in df.columns or 'destination' not in df.columns:
        # create repeating city pairs for edges
        n = len(df)
        srcs = []
        dsts = []
        for i in range(n):
            src = CITY_NAMES[i % len(CITY_NAMES)]
            dst = CITY_NAMES[(i+1) % len(CITY_NAMES)]
            srcs.append(src)
            dsts.append(dst)
        df['source'] = df.get('source', pd.Series(srcs))
        df['destination'] = df.get('destination', pd.Series(dsts))
    if 'edge_id' not in df.columns:
        df['edge_id'] = df['source'].astype(str) + "-" + df['destination'].astype(str)
    
    # Handle timestamp - only parse if column exists
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['dow'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
        except Exception:
            # If timestamp parsing fails, create default values
            df['timestamp'] = pd.Timestamp.now()
            df['hour'] = 0
            df['dow'] = 0
            df['is_weekend'] = 0
    else:
        df['timestamp'] = pd.Timestamp.now()
        df['hour'] = 0
        df['dow'] = 0
        df['is_weekend'] = 0

    if 'vehicle_type' in df.columns:
        le = LabelEncoder()
        df['vehicle_type_le'] = le.fit_transform(df['vehicle_type'].astype(str))
        joblib.dump(le, MODELS_DIR / "labelenc_vehicle_type.pkl")
    else:
        df['vehicle_type_le'] = 0

    if 'distance' not in df.columns:
        # if lat/lon present compute Haversine, else default small distance
        if {'lat_start','lon_start','lat_end','lon_end'}.issubset(df.columns):
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                phi1 = math.radians(lat1); phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            df['distance'] = df.apply(lambda r: haversine(r['lat_start'], r['lon_start'], r['lat_end'], r['lon_end']), axis=1)
        else:
            # FIX: Check if 'length' column exists and handle it properly
            if 'length' in df.columns:
                df['distance'] = df['length'].fillna(1.0)
            else:
                df['distance'] = 1.0  # Default distance if no length column

    if 'speed' not in df.columns:
        if {'distance','travel_time'}.issubset(df.columns):
            df['speed'] = df['distance'] / (df['travel_time'].replace(0, np.nan))
            df['speed'] = df['speed'].fillna(df['speed'].median())
        else:
            df['speed'] = 50.0

    if 'traffic_factor' not in df.columns:
        df['traffic_factor'] = 1.0

    df = df.sort_values('timestamp') if 'timestamp' in df.columns else df
    grp = df.groupby('edge_id')
    df['speed_r5m'] = grp['speed'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['traffic_factor_r5m'] = grp['traffic_factor'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df = df.fillna(method='ffill').fillna(0.0)
    return df

# -----------------------
# Training functions
# -----------------------
def run_train_pipeline(csv_path: str = CSV_PRIMARY, vanet_path: Optional[str] = None) -> Dict[str, Any]:
    st.info(f"Starting training pipeline using {csv_path} (vanet={vanet_path})")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # FIX: Check if timestamp column exists before parsing
    df_sample = pd.read_csv(csv_path, nrows=1)
    if 'timestamp' in df_sample.columns:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    else:
        df = pd.read_csv(csv_path)
        st.warning("No 'timestamp' column found in primary CSV. Using current time as default.")
    
    # FIX: Handle VANET CSV timestamp parsing safely
    vanet_df = None
    if vanet_path and os.path.exists(vanet_path):
        try:
            vanet_sample = pd.read_csv(vanet_path, nrows=1)
            if 'timestamp' in vanet_sample.columns:
                vanet_df = pd.read_csv(vanet_path, parse_dates=['timestamp'])
            else:
                vanet_df = pd.read_csv(vanet_path)
                st.warning("No 'timestamp' column found in VANET CSV. Using current time as default.")
        except Exception as e:
            st.warning(f"Could not load VANET CSV: {e}")
            vanet_df = None
    
    df = merge_primary_and_vanet(df, vanet_df)
    df = prepare_features(df)

    # ensure city coords columns exist for visualization (fill from city table)
    def ensure_coords(row):
        s = row['source']; d = row['destination']
        lat_s, lon_s = INDIAN_CITIES.get(s, (12.95,77.55))
        lat_d, lon_d = INDIAN_CITIES.get(d, (12.96,77.56))
        return pd.Series({'lat_start': lat_s, 'lon_start': lon_s, 'lat_end': lat_d, 'lon_end': lon_d})
    coords = df.apply(ensure_coords, axis=1)
    df['lat_start'] = coords['lat_start']; df['lon_start'] = coords['lon_start']
    df['lat_end'] = coords['lat_end']; df['lon_end'] = coords['lon_end']

    # target co2
    if 'co2_grams' in df.columns:
        df['target_co2'] = df['co2_grams']
    else:
        base_emission = 120.0
        congestion_sensitivity = 0.8
        df['target_co2'] = df['distance'].fillna(1.0) * (base_emission + congestion_sensitivity * df['traffic_factor'].fillna(1.0) / df['speed'].replace(0, np.nan).fillna(50.0))

    # CO2 model
    co2_feats = ['distance','speed','vehicle_type_le','hour','dow','is_weekend','traffic_factor','speed_r5m','traffic_factor_r5m']
    co2_feats = [c for c in co2_feats if c in df.columns]
    X = df[co2_feats].fillna(-1.0)
    y = df['target_co2']
    split_idx = int(0.8 * len(df)) if len(df) > 10 else int(0.7 * len(df))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {'objective':'regression','metric':['rmse','mae'],'learning_rate':0.05,'num_leaves':31,'seed':RANDOM_SEED,'verbosity':-1}
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train,lgb_val], early_stopping_rounds=30, verbose_eval=False)
    yp = gbm.predict(X_val)
    co2_rmse = float(np.sqrt(mean_squared_error(y_val, yp))) if len(y_val)>0 else None
    co2_mae = float(mean_absolute_error(y_val, yp)) if len(y_val)>0 else None
    gbm.save_model(str(MODELS_DIR / "lgb_co2.txt"))
    joblib.dump(co2_feats, MODELS_DIR / "feature_columns.pkl")

    # Traffic baseline
    if 'traffic_factor' in df.columns:
        traf_df = df[['traffic_factor','hour']].fillna(1.0) if 'hour' in df.columns else df[['traffic_factor']].fillna(1.0)
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        try:
            rf.fit(traf_df, df['traffic_factor'])
            traf_mae = float(mean_absolute_error(df['traffic_factor'], rf.predict(traf_df)))
            joblib.dump(rf, MODELS_DIR / "rf_traffic_forecast.joblib")
        except Exception:
            rf = None
            traf_mae = None
    else:
        rf = None; traf_mae = None

    # Security model
    sec_features = []
    for c in ['gps_integrity_score','packet_loss','signal_strength','gps_jump_flag']:
        if c in df.columns:
            sec_features.append(c)
    if not sec_features:
        sec_features = ['speed','traffic_factor']
    sec_X = df[sec_features].fillna(method='ffill').fillna(0.0)
    scaler = StandardScaler()
    sec_Xs = scaler.fit_transform(sec_X)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
    iso.fit(sec_Xs)
    joblib.dump(iso, MODELS_DIR / "iso_security.joblib")
    joblib.dump(scaler, MODELS_DIR / "security_scaler.joblib")

    # meta
    meta = {
        'version': '1.0',
        'created_at': datetime.datetime.now().isoformat(),
        'models': {
            'co2_lightgbm': {'path': str((MODELS_DIR / "lgb_co2.txt").resolve()), 'features': co2_feats, 'type':'regression'},
            'traffic_forecast_rf': {'path': str((MODELS_DIR / "rf_traffic_forecast.joblib").resolve()) if rf is not None else None, 'type':'regression'},
            'security_iso': {'path': str((MODELS_DIR / "iso_security.joblib").resolve()), 'scaler_path': str((MODELS_DIR / "security_scaler.joblib").resolve()), 'type':'anomaly_detection'}
        },
        'data_statistics': {'total_samples': int(len(df)), 'unique_edges': int(df['edge_id'].nunique())},
        'feature_weights': DEFAULT_WEIGHTS
    }
    sig = sign_dict(meta)
    meta['signature'] = sig
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    if not ADAPTIVE_PATH.exists():
        save_adaptive_weights(DEFAULT_WEIGHTS.copy())

    metrics = {'co2_rmse': co2_rmse, 'co2_mae': co2_mae, 'traffic_mae': traf_mae}
    return metrics

# -----------------------
# Model loader
# -----------------------
def _load_models():
    if not META_PATH.exists():
        raise FileNotFoundError("meta.json not found; train models first.")
    with open(META_PATH,'r') as f:
        meta = json.load(f)
    sig = meta.pop('signature', None)
    meta_copy = meta.copy()
    meta['signature'] = sig
    valid = True
    try:
        valid = verify_signature(meta_copy, sig) if sig else False
    except Exception:
        valid = False
    models = {'meta': meta, 'meta_signature_valid': valid}
    models['co2'] = lgb.Booster(model_file=meta['models']['co2_lightgbm']['path'])
    try:
        if meta['models'].get('traffic_forecast_rf',{}).get('path'):
            models['rf'] = joblib.load(meta['models']['traffic_forecast_rf']['path'])
        else:
            models['rf'] = None
    except:
        models['rf'] = None
    models['iso'] = joblib.load(meta['models']['security_iso']['path'])
    models['scaler'] = joblib.load(meta['models']['security_iso']['scaler_path'])
    models['co2_features'] = joblib.load(MODELS_DIR / "feature_columns.pkl")
    models['adaptive_weights'] = load_adaptive_weights()
    return models

# -----------------------
# Route prediction API
# -----------------------
def predict_route_api(start: str, end: str, vehicle_type: str = 'car', csv_path: str = CSV_PRIMARY) -> Dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV not found: " + csv_path)
    
    # FIX: Check if timestamp column exists before parsing
    df_sample = pd.read_csv(csv_path, nrows=1)
    if 'timestamp' in df_sample.columns:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    else:
        df = pd.read_csv(csv_path)
    
    df = prepare_features(df)

    # ensure coordinates are present (fill from city table)
    def ensure_coords_row(r):
        s = r['source']; d = r['destination']
        lat_s, lon_s = INDIAN_CITIES.get(s, (12.95,77.55))
        lat_d, lon_d = INDIAN_CITIES.get(d, (12.96,77.56))
        return pd.Series({'lat_start': lat_s, 'lon_start': lon_s, 'lat_end': lat_d, 'lon_end': lon_d})
    coords = df.apply(ensure_coords_row, axis=1)
    df['lat_start'] = coords['lat_start']; df['lon_start'] = coords['lon_start']
    df['lat_end'] = coords['lat_end']; df['lon_end'] = coords['lon_end']

    # Build directed graph from source/destination
    G = nx.DiGraph()
    # Use distinct city names as graph nodes
    base_edges = df.groupby(['source','destination']).agg({'distance':'mean'}).reset_index()
    for _,row in base_edges.iterrows():
        u=str(row['source']); v=str(row['destination']); d=float(row['distance'])
        G.add_edge(u,v, distance=d, edge_id=f"{u}-{v}")

    models = _load_models()
    co2_model = models['co2']; rf = models['rf']; iso = models['iso']; scaler = models['scaler']
    co2_feats = models['co2_features']
    adaptive_weights = models['adaptive_weights']
    a = adaptive_weights.get('co2_weight', DEFAULT_WEIGHTS['co2_weight'])
    b = adaptive_weights.get('traffic_weight', DEFAULT_WEIGHTS['traffic_weight'])
    c = adaptive_weights.get('security_weight', DEFAULT_WEIGHTS['security_weight'])

    edges_data = []
    # compute latest aggregated stats per edge (use most recent row per source-destination)
    for (u,v), group in df.groupby(['source','destination']):
        eid = f"{u}-{v}"
        row = group.sort_values('timestamp').iloc[-1:].copy()
        # ensure feature presence
        for ccol in ['distance','speed','vehicle_type_le','hour','dow','is_weekend','traffic_factor','speed_r5m','traffic_factor_r5m']:
            if ccol not in row.columns:
                row[ccol] = 0
        feat_row = row[co2_feats].fillna(-1.0)
        try:
            co2_pred = float(co2_model.predict(feat_row)[0])
        except Exception:
            co2_pred = float(row.get('target_co2', row.get('distance',1.0) * 120.0))
        dist = float(row.get('distance', 1.0))
        spd = float(row.get('speed', 50.0))
        travel_time = (dist / max(0.1, spd)) * 3600.0
        # security features
        sec_feats = []
        for sf in ['gps_integrity_score','packet_loss','signal_strength','gps_jump_flag','packet_loss_mean']:
            if sf in row.columns:
                sec_feats.append(float(row[sf].iloc[0]))
        if not sec_feats:
            sec_feats = [float(row.get('speed',50.0).iloc[0]), float(row.get('traffic_factor',1.0).iloc[0])]
        sec_feat = np.array(sec_feats).reshape(1,-1)
        try:
            sec_scaled = scaler.transform(sec_feat)
            sec_raw = float(iso.decision_function(sec_scaled)[0])
            sec_norm = 100.0 / (1.0 + math.exp(-sec_raw))
            sec_score = float(np.clip(sec_norm, 0.0, 100.0))
        except Exception:
            sec_score = 50.0
        # coords
        lat_s = float(row.get('lat_start', INDIAN_CITIES.get(u, (12.95,77.55))[0]).iloc[0])
        lon_s = float(row.get('lon_start', INDIAN_CITIES.get(u, (12.95,77.55))[1]).iloc[0])
        lat_e = float(row.get('lat_end', INDIAN_CITIES.get(v, (12.96,77.56))[0]).iloc[0])
        lon_e = float(row.get('lon_end', INDIAN_CITIES.get(v, (12.96,77.56))[1]).iloc[0])

        edges_data.append({
            'u': u, 'v': v, 'edge_id': eid,
            'distance': dist, 'speed': spd,
            'pred_co2': co2_pred, 'pred_time': travel_time,
            'security_score': sec_score,
            'lat_start': lat_s, 'lon_start': lon_s, 'lat_end': lat_e, 'lon_end': lon_e
        })

    # compute weights
    for e in edges_data:
        security_penalty = (100.0 - e['security_score'])
        e['weight'] = a * e['pred_co2'] + b * (e['pred_time'] / 60.0) + c * (security_penalty)

    # insert into graph meta
    for e in edges_data:
        if G.has_edge(e['u'], e['v']):
            G[e['u']][e['v']]['weight'] = float(e['weight'])
            G[e['u']][e['v']]['meta'] = e

    # find path by city names
    try:
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        path_edges = []
        tot_co2 = 0.0; tot_time = 0.0; secs = []
        for i in range(len(path)-1):
            u = path[i]; v = path[i+1]
            dat = G[u][v].get('meta', None)
            if dat is None:
                dat = {'edge_id': f"{u}-{v}", 'pred_co2':0.0, 'pred_time':0.0, 'security_score':50.0}
            path_edges.append({
                'edge_id': dat.get('edge_id'),
                'from': u, 'to': v,
                'co2': dat.get('pred_co2',0.0),
                'time': dat.get('pred_time',0.0),
                'security': dat.get('security_score',50.0),
                'lat_start': dat.get('lat_start'), 'lon_start': dat.get('lon_start'),
                'lat_end': dat.get('lat_end'), 'lon_end': dat.get('lon_end')
            })
            tot_co2 += dat.get('pred_co2',0.0); tot_time += dat.get('pred_time',0.0); secs.append(dat.get('security_score',50.0))
    except Exception:
        path_edges = []
        tot_co2 = 0.0; tot_time = 0.0; secs = []
        if edges_data:
            ed = edges_data[0]
            path_edges.append({
                'edge_id': ed['edge_id'], 'from': ed['u'], 'to': ed['v'],
                'co2': ed['pred_co2'], 'time': ed['pred_time'], 'security': ed['security_score'],
                'lat_start': ed['lat_start'], 'lon_start': ed['lon_start'],
                'lat_end': ed['lat_end'], 'lon_end': ed['lon_end']
            })
            tot_co2 += ed['pred_co2']; tot_time += ed['pred_time']; secs.append(ed['security_score'])

    route = {
        'edges': path_edges,
        'estimated_co2': float(tot_co2),
        'estimated_travel_time': float(tot_time),
        'security_score': float(sum(secs)/len(secs)) if secs else 50.0,
        'used_weights': adaptive_weights,
        'meta_signature_valid': models.get('meta_signature_valid', False)
    }
    return route

# -----------------------
# Streamlit UI
# -----------------------
st.title("EcoRoute — Carbon-Aware Cyber-Secure Route Optimization (India demo)")
st.markdown("Upload datasets (primary + optional VANET), train models, and visualize eco-secure routes between Indian cities.")

# Sidebar uploads
st.sidebar.header("Datasets upload")
uploaded_primary = st.sidebar.file_uploader("Primary CSV (traffic/co2)", type=['csv'])
uploaded_vanet = st.sidebar.file_uploader("Optional VANET CSV", type=['csv'])
if uploaded_primary:
    p = UPLOAD_DIR / "uploaded_primary.csv"
    with open(p,'wb') as f:
        f.write(uploaded_primary.getbuffer())
    st.sidebar.success(f"Saved {p.name}")
    CSVp = str(p)
else:
    CSVp = CSV_PRIMARY if os.path.exists(CSV_PRIMARY) else None

if uploaded_vanet:
    v = UPLOAD_DIR / "uploaded_vanet.csv"
    with open(v,'wb') as f:
        f.write(uploaded_vanet.getbuffer())
    st.sidebar.success(f"Saved {v.name}")
    CSVv = str(v)
else:
    CSVv = CSV_VANET if os.path.exists(CSV_VANET) else None

st.sidebar.markdown("---")
if st.sidebar.button("Train models"):
    if CSVp is None:
        st.error("Primary CSV not found or uploaded.")
    else:
        with st.spinner("Training models... this can take a few minutes"):
            try:
                metrics = run_train_pipeline(csv_path=CSVp, vanet_path=CSVv)
                st.success("Training complete.")
                st.json(metrics)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

# Predict controls
st.header("Predict Eco-Secure Route (Cities)")
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    start_input = st.selectbox("Start city", options=CITY_NAMES, index=0)
with col2:
    end_input = st.selectbox("End city", options=CITY_NAMES, index=1)
with col3:
    vehicle_choice = st.selectbox("Vehicle type", options=["car","truck","ev","motorcycle"])
with col4:
    metric_choice = st.selectbox("Map metric", options=["security","co2","time"])

# Animation toggle & dominance factor
animate_checkbox = st.sidebar.checkbox("Animate vehicle-dominant edges", value=True)
VEHICLE_DOM_FACTOR = st.sidebar.slider("Vehicle dominance factor (vehicle_co2 >= factor * max(other))", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if st.button("Get Route"):
    try:
        if CSVp is None:
            st.error("Primary CSV not found — upload or place file.")
        else:
            route = predict_route_api(start=start_input, end=end_input, vehicle_type=vehicle_choice, csv_path=CSVp)
            st.success("Route computed.")
            st.write(f"Estimated CO₂ (g): **{route['estimated_co2']:.2f}**")
            st.write(f"Estimated travel time (s): **{route['estimated_travel_time']:.2f}**")
            st.write(f"Route security score (0-100): **{route['security_score']:.1f}**")

            edges_df = pd.DataFrame(route['edges'])
            # ensure emission component columns exist and fallbacks
            edges_df['vehicle_co2'] = edges_df.get('vehicle_co2', edges_df.get('co2', edges_df.get('pred_co2', 0.0))).astype(float)
            edges_df['grid_co2'] = edges_df.get('grid_co2', 0.0).astype(float)
            edges_df['server_co2'] = edges_df.get('server_co2', 0.0).astype(float)
            edges_df['upstream_co2'] = edges_df.get('upstream_co2', 0.0).astype(float)
            if 'total_co2' not in edges_df.columns:
                edges_df['total_co2'] = edges_df.apply(lambda r: float(r.get('vehicle_co2',0.0)) + float(r.get('grid_co2',0.0)) + float(r.get('server_co2',0.0)) + float(r.get('upstream_co2',0.0)), axis=1)

            st.dataframe(edges_df[['from','to','edge_id','total_co2','vehicle_co2','time','security']])

            # Map
            avg_lat = float(np.mean(edges_df['lat_start'].tolist() + edges_df['lat_end'].tolist()))
            avg_lon = float(np.mean(edges_df['lon_start'].tolist() + edges_df['lon_end'].tolist()))
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

            for _, r in edges_df.iterrows():
                lat1, lon1 = float(r['lat_start']), float(r['lon_start'])
                lat2, lon2 = float(r['lat_end']), float(r['lon_end'])

                vehicle_co2 = float(r.get('vehicle_co2', 0.0))
                grid_co2 = float(r.get('grid_co2', 0.0))
                server_co2 = float(r.get('server_co2', 0.0))
                upstream_co2 = float(r.get('upstream_co2', 0.0))
                total_co2 = float(r.get('total_co2', vehicle_co2 + grid_co2 + server_co2 + upstream_co2))
                travel_time = float(r.get('time', r.get('pred_time', 0.0)))
                security = float(r.get('security', r.get('security_score', 50.0)))

                max_other = max(grid_co2, server_co2, upstream_co2)
                is_vehicle_dominant = vehicle_co2 >= (VEHICLE_DOM_FACTOR * max_other)

                tooltip = (
                    f"Edge {r.get('edge_id')} ({r.get('from')} → {r.get('to')})\n"
                    f"total CO₂: {total_co2:.1f} g\n"
                    f"(vehicle: {vehicle_co2:.1f} g, grid: {grid_co2:.1f} g, server: {server_co2:.1f} g)\n"
                    f"Time: {travel_time:.1f}s | Sec: {security:.1f}"
                )

                if is_vehicle_dominant and animate_checkbox:
                    AntPath(
                        locations=[(lat1, lon1), (lat2, lon2)],
                        reverse=False,
                        delay=1500,
                        dash_array=[10, 20],
                        color='crimson',
                        weight=6,
                        pulse_color='yellow',
                        tooltip=tooltip
                    ).add_to(m)
                    folium.CircleMarker(location=(lat1, lon1),
                                        radius=4,
                                        color='black',
                                        fill=True,
                                        fill_color='white',
                                        fill_opacity=1.0).add_to(m)
                else:
                    severity = max(grid_co2, server_co2)
                    if severity <= 50:
                        color = 'green'
                    elif severity <= 200:
                        color = 'orange'
                    else:
                        color = 'purple'
                    folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],
                                    color=color, weight=4, opacity=0.7, tooltip=tooltip).add_to(m)
                    mid_lat = (lat1 + lat2) / 2.0
                    mid_lon = (lon1 + lon2) / 2.0
                    radius = max(4, min(30, math.sqrt(severity + 1)))
                    folium.CircleMarker(location=(mid_lat, mid_lon),
                                        radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.6,
                                        tooltip=f"Infra CO₂ hotspot: {severity:.1f} g").add_to(m)

            # city markers for start and end
            try:
                s = start_input; e = end_input
                lat_s, lon_s = INDIAN_CITIES.get(s, (12.95,77.55))
                lat_e, lon_e = INDIAN_CITIES.get(e, (12.96,77.56))
                folium.Marker(location=[lat_s,lon_s], popup=f"Start: {s}", icon=folium.Icon(color='blue', icon='play')).add_to(m)
                folium.Marker(location=[lat_e,lon_e], popup=f"End: {e}", icon=folium.Icon(color='red', icon='flag')).add_to(m)
            except Exception:
                pass

            st.subheader("Map Visualization (animated vehicle edges vs infra hotspots)")
            st.write("Toggle 'Animate vehicle-dominant edges' in the sidebar to enable/disable animation.")
            st_folium(m, width=1000, height=650)

            # feedback & adaptive weights
            st.markdown("---")
            st.write("Feedback (optional): did the route meet expectations?")
            fb_col1, fb_col2 = st.columns([1,1])
            with fb_col1:
                ok = st.button("Yes — route OK")
            with fb_col2:
                not_ok = st.button("No — route had problems")
            if ok or not_ok:
                prev = load_adaptive_weights()
                feedback = {}
                feedback['security_violation'] = bool(not_ok)
                new_weights = adapt_weights_on_feedback(prev, feedback)
                st.success(f"Adaptive weights updated: {new_weights}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

# show model files
st.header("Model artifacts (models_output/)")
files = list(MODELS_DIR.glob("*"))
if files:
    for f in files:
        st.write(f"- {f.name}  ({f.stat().st_size} bytes)")
else:
    st.write("No model artifacts found. Train models to generate them.")

st.markdown("---")
st.markdown("**Notes:** This app is designed for local/demo use. For production: secure keys, use TLS, and run training in background workers.")
