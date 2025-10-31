# train_final.py - Final training script with proper class definition
import os
import pandas as pd
import numpy as np
import joblib
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Define the ensemble predictor class FIRST
class EnsemblePredictor:
    def __init__(self, nn_model, rf_model, ridge_model, scaler, feature_cols, weights=[0.3, 0.4, 0.3]):
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.ridge_model = ridge_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.weights = weights
        
    def predict(self, input_df):
        # Ensure columns present
        missing = [c for c in self.feature_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Prepare features
        X_in = input_df[self.feature_cols].values
        X_in_scaled = self.scaler.transform(X_in)
        
        # Get predictions from all models
        nn_pred = self.nn_model.predict(X_in_scaled)
        rf_pred = self.rf_model.predict(X_in_scaled)
        ridge_pred = self.ridge_model.predict(X_in_scaled)
        
        # Weighted ensemble
        ensemble_pred = (self.weights[0] * nn_pred + 
                        self.weights[1] * rf_pred + 
                        self.weights[2] * ridge_pred)
        
        # Clip to reasonable credit score range
        ensemble_pred = np.clip(ensemble_pred, 300.0, 850.0)
        return ensemble_pred

# Configuration
CSV_PATH = r"/Users/shwetank/Desktop/kannu"
MODEL_DIR = r"/Users/shwetank/Desktop/kannu"
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_predictor.pkl")

print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Simple preprocessing
df_encoded = df.copy()
if 'CAT_GAMBLING' in df_encoded.columns:
    gambling_mapping = {'High': 2, 'No': 0, 'Low': 1}
    df_encoded['CAT_GAMBLING'] = df_encoded['CAT_GAMBLING'].map(gambling_mapping).fillna(0)

# Features and target
feature_cols = [c for c in df_encoded.columns if c not in ['CUST_ID', 'CREDIT_SCORE', 'DEFAULT']]
X = df_encoded[feature_cols].values
y = df_encoded['CREDIT_SCORE'].values

# Train/test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")
# Train simple models
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Simple neural network-like model (using Ridge as placeholder)
nn_model = Ridge(alpha=0.1)
nn_model.fit(X_train_scaled, y_train)

# Create ensemble predictor
ensemble_predictor = EnsemblePredictor(nn_model, rf_model, ridge_model, scaler, feature_cols)
joblib.dump(ensemble_predictor, ENSEMBLE_MODEL_PATH)

# Save supporting files
with open(os.path.join(MODEL_DIR, "feature_cols.json"), 'w') as f:
    json.dump(feature_cols, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# Test the model
test_sample = df_encoded[feature_cols].iloc[:1]
prediction = ensemble_predictor.predict(test_sample)[0]
print(f"Sample prediction: {prediction:.1f}")

print("✓ Models trained and saved successfully!")
print(f"Ensemble model saved to: {ENSEMBLE_MODEL_PATH}")