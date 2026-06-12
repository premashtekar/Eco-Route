# save as train_cybersecurity_eco.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Config
# -------------------------
CSV_PATH = "carbon_aware_cybersecurity_dataset.csv"
OUT_DIR = "cybersecurity_models"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEED = 42

# -------------------------
# Data Loading for Cybersecurity
# -------------------------
def load_cybersecurity_data(file_path):
    """Load and prepare cybersecurity dataset"""
    df = pd.read_csv(file_path)
    print(f"✅ Cybersecurity data loaded: {len(df)} records, {len(df.columns)} features")

    # Basic validation
    required_cols = ['power_consumption_watts', 'carbon_emission_gCO2eq', 'status']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")

    print(f"📊 Data summary:")
    print(f"   - Normal records: {len(df[df['status']=='normal')}")
    print(f"   - Anomaly records: {len(df[df['status']=='anomaly')}")
    print(f"   - Avg power consumption: {df['power_consumption_watts'].mean():.2f}W")
    print(f"   - Avg carbon emission: {df['carbon_emission_gCO2eq'].mean():.2f}gCO2eq")

    return df

# Load data
df = load_cybersecurity_data(CSV_PATH)

# -------------------------
# Feature Engineering for Cybersecurity
# -------------------------
class CybersecurityFeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.scalers = {}

    def create_core_features(self, df):
        """Create features for cybersecurity carbon optimization"""

        # 1. Protocol encoding
        if 'protocol_type' in df.columns:
            self.encoders['protocol'] = LabelEncoder()
            df['protocol_encoded'] = self.encoders['protocol'].fit_transform(df['protocol_type'])
            print(f"🔗 Protocols: {list(self.encoders['protocol'].classes_)}")

        # 2. Connection state encoding
        if 'connection_state' in df.columns:
            self.encoders['connection'] = LabelEncoder()
            df['connection_encoded'] = self.encoders['connection'].fit_transform(df['connection_state'])
            print(f"🔌 Connection states: {list(self.encoders['connection'].classes_)}")

        # 3. Resource utilization features
        resource_cols = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']
        for col in resource_cols:
            if col in df.columns:
                # Create efficiency ratio
                df[f'{col}_efficiency'] = df['packet_count'] / (df[col].replace(0, 0.1))

        # 4. Network efficiency metrics
        if {'packet_count', 'byte_count'}.issubset(df.columns):
            df['avg_packet_size'] = df['byte_count'] / df['packet_count'].replace(0, 1)
            df['bytes_per_packet_ratio'] = df['byte_count'] / df['packet_count'].replace(0, 1)

        # 5. Carbon intensity features
        if {'power_consumption_watts', 'carbon_emission_gCO2eq'}.issubset(df.columns):
            df['carbon_efficiency'] = df['packet_count'] / (df['carbon_emission_gCO2eq'].replace(0, 0.1))
            df['power_efficiency'] = df['packet_count'] / (df['power_consumption_watts'].replace(0, 0.1))

        # 6. Entropy-based features (for security)
        if 'payload_entropy' in df.columns:
            df['high_entropy'] = (df['payload_entropy'] > df['payload_entropy'].median()).astype(int)

        return df

    def create_temporal_features(self, df):
        """Create time-based patterns (if timestamp exists)"""
        # If you had timestamps, you'd add hour, day patterns here
        # For now, create synthetic time patterns based on index
        df['synthetic_hour'] = (df.index % 24)
        df['synthetic_day_part'] = pd.cut(df['synthetic_hour'],
                                       bins=[0, 6, 12, 18, 24],
                                       labels=['night', 'morning', 'afternoon', 'evening'])

        return df

# Apply feature engineering
feature_engineer = CybersecurityFeatureEngineer()
df = feature_engineer.create_core_features(df)
df = feature_engineer.create_temporal_features(df)

# -------------------------
# Carbon Emission Prediction Model
# -------------------------
def train_carbon_model(df):
    """Train model to predict carbon emissions from network activity"""

    # Select features for carbon prediction
    carbon_features = [
        'packet_count', 'byte_count', 'flow_duration', 'avg_pkt_size',
        'cpu_usage', 'memory_usage', 'disk_io', 'network_io', 'vm_count',
        'protocol_encoded', 'connection_encoded'
    ]

    # Add any engineered features
    carbon_features += [col for col in df.columns if col.endswith('_efficiency')]
    carbon_features = [col for col in carbon_features if col in df.columns]

    print(f"🔧 Using {len(carbon_features)} features for carbon prediction:")
    print(f"   Features: {carbon_features}")

    X = df[carbon_features].fillna(0)
    y = df['carbon_emission_gCO2eq']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'verbosity': -1,
        'seed': RANDOM_SEED,
    }

    # Train model
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    print("🌳 Training Carbon Emission Predictor...")
    carbon_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_test],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Evaluate
    y_pred = carbon_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ Carbon Model Performance:")
    print(f"   - RMSE: {rmse:.2f} gCO2eq")
    print(f"   - MAE: {mae:.2f} gCO2eq")
    print(f"   - Actual mean: {y_test.mean():.2f} gCO2eq")

    return carbon_model, carbon_features

carbon_model, carbon_features = train_carbon_model(df)

# -------------------------
# Anomaly Detection Model
# -------------------------
def train_anomaly_detection(df):
    """Train anomaly detection using the 'status' column as ground truth"""

    # Use existing labeled anomalies
    X = df.drop(['status'], axis=1)

    # Select numeric features for anomaly detection
    anomaly_features = [
        'packet_count', 'byte_count', 'flow_duration', 'avg_pkt_size',
        'payload_entropy', 'cpu_usage', 'memory_usage', 'disk_io',
        'network_io', 'vm_count', 'power_consumption_watts'
    ]
    anomaly_features = [col for col in anomaly_features if col in X.columns]

    # Add encoded features
    anomaly_features += [col for col in X.columns if col.endswith('_encoded')]

    print(f"🛡️ Using {len(anomaly_features)} features for anomaly detection")

    X_anomaly = X[anomaly_features].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)

    # Convert status to binary (already labeled in your data)
    y_anomaly = (df['status'] == 'anomaly').astype(int)

    print(f"🎯 Anomaly distribution: {y_anomaly.value_counts().to_dict()}")

    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=len(df[df['status']=='anomaly']) / len(df),  # Use actual anomaly rate
        random_state=RANDOM_SEED,
        verbose=1
    )

    iso_forest.fit(X_scaled)

    # Also train a supervised classifier for comparison
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_anomaly, test_size=0.2, random_state=RANDOM_SEED, stratify=y_anomaly
    )

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        verbose=1
    )
    rf_classifier.fit(X_train, y_train)

    # Evaluate supervised model
    y_pred = rf_classifier.predict(X_test)
    print(f"✅ Supervised Anomaly Detection Performance:")
    print(classification_report(y_test, y_pred))

    return iso_forest, rf_classifier, scaler, anomaly_features

iso_model, rf_model, anomaly_scaler, anomaly_features = train_anomaly_detection(df)

# -------------------------
# Power Consumption Model
# -------------------------
def train_power_model(df):
    """Train model to predict power consumption"""

    power_features = [
        'packet_count', 'byte_count', 'cpu_usage', 'memory_usage',
        'disk_io', 'network_io', 'vm_count', 'carbon_emission_gCO2eq'
    ]
    power_features = [col for col in power_features if col in df.columns]

    X = df[power_features].fillna(0)
    y = df['power_consumption_watts']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'seed': RANDOM_SEED,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test)

    print("⚡ Training Power Consumption Predictor...")
    power_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_test],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    y_pred = power_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"✅ Power Model Performance - RMSE: {rmse:.2f}W")

    return power_model, power_features

power_model, power_features = train_power_model(df)

# -------------------------
# Save Models & Metadata
# -------------------------
def save_models_and_metadata():
    """Save all models and metadata"""

    # Save models
    carbon_model.save_model(os.path.join(OUT_DIR, 'carbon_predictor.txt'))
    joblib.dump(iso_model, os.path.join(OUT_DIR, 'anomaly_detector_iso.joblib'))
    joblib.dump(rf_model, os.path.join(OUT_DIR, 'anomaly_detector_rf.joblib'))
    joblib.dump(anomaly_scaler, os.path.join(OUT_DIR, 'anomaly_scaler.joblib'))
    power_model.save_model(os.path.join(OUT_DIR, 'power_predictor.txt'))

    # Save encoders
    for name, encoder in feature_engineer.encoders.items():
        joblib.dump(encoder, os.path.join(OUT_DIR, f'encoder_{name}.pkl'))

    # Create metadata
    metadata = {
        'version': '1.0',
        'domain': 'cybersecurity_carbon_optimization',
        'created_at': datetime.datetime.now().isoformat(),
        'models': {
            'carbon_predictor': {
                'features': carbon_features,
                'purpose': 'predict_carbon_emissions_from_network_activity'
            },
            'anomaly_detector_iso': {
                'features': anomaly_features,
                'type': 'unsupervised_anomaly_detection'
            },
            'anomaly_detector_rf': {
                'features': anomaly_features,
                'type': 'supervised_anomaly_classification'
            },
            'power_predictor': {
                'features': power_features,
                'purpose': 'predict_power_consumption'
            }
        },
        'data_statistics': {
            'total_samples': len(df),
            'normal_count': len(df[df['status'] == 'normal']),
            'anomaly_count': len(df[df['status'] == 'anomaly']),
            'avg_carbon_emission': float(df['carbon_emission_gCO2eq'].mean()),
            'avg_power_consumption': float(df['power_consumption_watts'].mean())
        }
    }

    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata

metadata = save_models_and_metadata()

# -------------------------
# Final Report
# -------------------------
print("\n" + "="*60)
print("🔒 CYBERSECURITY CARBON OPTIMIZATION - TRAINING COMPLETE")
print("="*60)
print(f"✅ Carbon Emission Model: {len(carbon_features)} features")
print(f"✅ Anomaly Detection: {len(anomaly_features)} features (supervised + unsupervised)")
print(f"✅ Power Consumption Model: {len(power_features)} features")
print(f"💾 Models saved to: {OUT_DIR}/")
print(f"📊 Dataset: {len(df)} records ({metadata['data_statistics']['anomaly_count']} anomalies)")
print("="*60)

# Test loading
def test_loading():
    print("\n🧪 Testing model loading...")
    try:
        lgb.Booster(model_file=os.path.join(OUT_DIR, 'carbon_predictor.txt'))
        joblib.load(os.path.join(OUT_DIR, 'anomaly_detector_iso.joblib'))
        joblib.load(os.path.join(OUT_DIR, 'anomaly_detector_rf.joblib'))
        print("🎉 All models loaded successfully!")
    except Exception as e:
        print(f"❌ Loading failed: {e}")

test_loading()
