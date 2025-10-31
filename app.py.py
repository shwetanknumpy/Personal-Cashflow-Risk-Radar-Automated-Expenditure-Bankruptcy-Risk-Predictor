# Personal Cashflow Risk Radar — Deep Learning + LLM + Flask REST API
# This notebook:
# 1. Loads and preprocesses the credit_score.csv dataset.
# 2. Builds and tunes machine learning models for credit score prediction.
# 3. Evaluates models and creates ensemble predictions.
# 4. Prepares for Flask API deployment with explanation capabilities.

# Install required packages
# import sys
# !{sys.executable} -m pip install --quiet tensorflow scikit-learn matplotlib pandas flask openai transformers torch joblib

# Standard libraries
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression

# Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Flask for REST API
from flask import Flask, request, jsonify

# Configuration
CSV_PATH = r"/Users/shwetank/Desktop/kannu"       
USER_ID_COL = "CUST_ID"
TARGET_COL = "CREDIT_SCORE"

import os

# Base directory where all model artifacts are stored
BASE_DIR = "/Users/shwetank/Desktop/kannu"

# Model artifact paths
MODEL_PATH = os.path.join(BASE_DIR, "credit_score_keras_best")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_META_PATH = os.path.join(BASE_DIR, "feature_cols.json")
RIDGE_MODEL_PATH = os.path.join(BASE_DIR, "ridge_model.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(BASE_DIR, "ensemble_predictor.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# Load and inspect dataset
print("Loading dataset...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at path: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target variable stats:")
print(f"  Min: {df[TARGET_COL].min()}, Max: {df[TARGET_COL].max()}, Mean: {df[TARGET_COL].mean():.2f}")
print("\nFirst 5 rows:")
print(df.head())

# Data preprocessing
print("\nPreprocessing data...")
df_encoded = df.copy()

# Convert categorical columns to numeric
if 'CAT_GAMBLING' in df_encoded.columns:
    gambling_mapping = {'High': 2, 'No': 0, 'Low': 1}
    df_encoded['CAT_GAMBLING'] = df_encoded['CAT_GAMBLING'].map(gambling_mapping)
    df_encoded['CAT_GAMBLING'] = df_encoded['CAT_GAMBLING'].fillna(0)

# Build feature columns (exclude grouping keys and target)
group_keys = [USER_ID_COL]
feature_cols = [c for c in df_encoded.columns if c not in group_keys + [TARGET_COL, "DEFAULT"]]

# Check for any remaining non-numeric columns
non_numeric_cols = []
for col in feature_cols:
    if df_encoded[col].dtype == 'object':
        non_numeric_cols.append(col)

if non_numeric_cols:
    print(f"Converting non-numeric columns: {non_numeric_cols}")
    for col in non_numeric_cols:
        unique_vals = df_encoded[col].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df_encoded[col] = df_encoded[col].map(mapping)
        df_encoded[col] = df_encoded[col].fillna(0)

# Extract features and target
X = df_encoded[feature_cols].astype(np.float32).values
y = df_encoded[TARGET_COL].astype(np.float32).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and feature columns
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
with open(FEATURES_META_PATH, "w") as f:
    json.dump(feature_cols, f)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Number of features: {len(feature_cols)}")
print(f"First 10 feature columns: {feature_cols[:10]}")

# Build and train neural network model
def build_model(hp):
    model = Sequential()
    
    # Input layer
    model.add(Dense(hp.Int("input_units", 64, 256, step=32), 
                   activation='relu', 
                   input_shape=(X_train_scaled.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float("input_dropout", 0.1, 0.5, step=0.1)))
    
    # Hidden layers
    for i in range(hp.Int("num_layers", 1, 3)):
        units = hp.Int(f"units_{i}", min_value=32, max_value=128, step=32)
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float(f"dropout_{i}", 0.1, 0.4, step=0.1)))
    
    # Output layer for regression
    model.add(Dense(1, activation='linear'))
    
    # Learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), 
                 loss='mse', 
                 metrics=['mae'])
    return model

# Hyperparameter tuning
print("\nStarting hyperparameter tuning...")
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='credit_score'
)

tuner.search_space_summary()

# Early stopping
es_tuner = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Run the search
tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.15, 
             callbacks=[es_tuner], batch_size=32, verbose=1)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate neural network
y_pred_nn = best_model.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_r2 = r2_score(y_test, y_pred_nn)

print(f"\nNeural Network Results:")
print(f"MSE: {nn_mse:.2f}")
print(f"R²: {nn_r2:.4f}")

# Train and evaluate Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest Results:")
print(f"MSE: {rf_mse:.2f}")
print(f"R²: {rf_r2:.4f}")

# Train and evaluate Ridge Regression
print("\nTraining Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression Results:")
print(f"MSE: {ridge_mse:.2f}")
print(f"R²: {ridge_r2:.4f}")

# Create ensemble predictions
print("\nCreating ensemble predictions...")
ensemble_pred = (0.3 * y_pred_nn + 0.3 * y_pred_rf + 0.4 * y_pred_ridge)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"Ensemble Results:")
print(f"MSE: {ensemble_mse:.2f}")
print(f"R²: {ensemble_r2:.4f}")

# Feature importance analysis
print("\nTop 10 most important features (Ridge coefficients):")
feature_importance = np.abs(ridge_model.coef_)
important_features_idx = np.argsort(feature_importance)[-10:][::-1]

for idx in important_features_idx:
    print(f"  {feature_cols[idx]}: {feature_importance[idx]:.4f}")

# Create ensemble predictor class
class EnsemblePredictor:
    def __init__(self, nn_model, rf_model, ridge_model, scaler, feature_cols, weights=[0.3, 0.3, 0.4]):
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
        X_in = input_df[self.feature_cols].astype(np.float32).values
        X_in_scaled = self.scaler.transform(X_in)
        
        # Get predictions from all models
        nn_pred = self.nn_model.predict(X_in_scaled).flatten()
        rf_pred = self.rf_model.predict(X_in_scaled)
        ridge_pred = self.ridge_model.predict(X_in_scaled)
        
        # Weighted ensemble
        ensemble_pred = (self.weights[0] * nn_pred + 
                        self.weights[1] * rf_pred + 
                        self.weights[2] * ridge_pred)
        
        # Clip to reasonable credit score range
        ensemble_pred = np.clip(ensemble_pred, 300.0, 850.0)
        return ensemble_pred

# Create and save ensemble predictor
ensemble_predictor = EnsemblePredictor(best_model, rf_model, ridge_model, scaler, feature_cols)
joblib.dump(ensemble_predictor, ENSEMBLE_MODEL_PATH)

# Save individual models
best_model.save(MODEL_PATH, include_optimizer=False)
joblib.dump(rf_model, RF_MODEL_PATH)
joblib.dump(ridge_model, RIDGE_MODEL_PATH)

print(f"\nModels saved successfully:")
print(f"- Neural Network: {MODEL_PATH}")
print(f"- Random Forest: {RF_MODEL_PATH}")
print(f"- Ridge Regression: {RIDGE_MODEL_PATH}")
print(f"- Ensemble: {ENSEMBLE_MODEL_PATH}")

# Explanation function
def explain_rule_based(pred_score, features_dict):
    """Enhanced explanation using feature importance"""
    lines = []
    
    # Interpret score
    if pred_score >= 720:
        lines.append(f"Score {pred_score:.0f} — Excellent Credit")
    elif pred_score >= 680:
        lines.append(f"Score {pred_score:.0f} — Good Credit")
    elif pred_score >= 620:
        lines.append(f"Score {pred_score:.0f} — Fair Credit") 
    else:
        lines.append(f"Score {pred_score:.0f} — Poor Credit")
    
    # Top contributing factors based on importance
    important_features = ['R_DEBT_INCOME', 'T_TAX_12', 'T_GROCERIES_12']
    lines.append("\nKey Factors:")
    
    for feat in important_features:
        if feat in features_dict:
            value = features_dict[feat]
            if feat == 'R_DEBT_INCOME':
                lines.append(f"- Debt-to-Income Ratio: {value:.1f}x income")
            elif 'TAX' in feat:
                lines.append(f"- Tax Payments: ${value:,.0f} annually")
            elif 'GROCERIES' in feat:
                lines.append(f"- Grocery Spending: ${value:,.0f} annually")
    
    # Actionable advice based on key factors
    lines.append("\nRecommendations:")
    if features_dict.get('R_DEBT_INCOME', 0) > 5:
        lines.append("1. Reduce outstanding debt")
    else:
        lines.append("1. Maintain current debt levels")
    
    lines.append("2. Continue timely bill payments")
    lines.append("3. Monitor credit utilization")
    
    return "\n".join(lines)

# Test the ensemble predictor
print("\nTesting ensemble predictor...")
test_sample = df_encoded[feature_cols].iloc[:1]
prediction = ensemble_predictor.predict(test_sample)[0]
print(f"Sample prediction: {prediction:.1f}")

# Create Flask app template
flask_app_code = '''
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models and preprocessing objects
try:
    ensemble_predictor = joblib.load(r"{ensemble_model_path}")
    print("Ensemble model loaded successfully")
except Exception as e:
    print(f"Error loading ensemble model: {{e}}")
    ensemble_predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({{"error": "No data provided"}}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        if ensemble_predictor:
            prediction = ensemble_predictor.predict(input_df)[0]
            
            # Generate explanation
            explanation = explain_rule_based(prediction, data)
            
            return jsonify({{
                "predicted_credit_score": float(prediction),
                "explanation": explanation
            }})
        else:
            return jsonify({{"error": "Model not loaded"}}), 500
            
    except Exception as e:
        return jsonify({{"error": str(e)}}), 400

def explain_rule_based(pred_score, features_dict):
    """Enhanced explanation using feature importance"""
    lines = []
    
    # Interpret score
    if pred_score >= 720:
        lines.append(f"Score {{pred_score:.0f}} — Excellent Credit")
    elif pred_score >= 680:
        lines.append(f"Score {{pred_score:.0f}} — Good Credit")
    elif pred_score >= 620:
        lines.append(f"Score {{pred_score:.0f}} — Fair Credit") 
    else:
        lines.append(f"Score {{pred_score:.0f}} — Poor Credit")
    
    # Top contributing factors
    important_features = ['R_DEBT_INCOME', 'T_TAX_12', 'T_GROCERIES_12']
    lines.append("\\\\nKey Factors:")
    
    for feat in important_features:
        if feat in features_dict:
            value = features_dict[feat]
            if feat == 'R_DEBT_INCOME':
                lines.append(f"- Debt-to-Income Ratio: {{value:.1f}}x income")
            elif 'TAX' in feat:
                lines.append(f"- Tax Payments: ${{value:,.0f}} annually")
            elif 'GROCERIES' in feat:
                lines.append(f"- Grocery Spending: ${{value:,.0f}} annually")
    
    # Actionable advice
    lines.append("\\\\nRecommendations:")
    if features_dict.get('R_DEBT_INCOME', 0) > 5:
        lines.append("1. Reduce outstanding debt")
    else:
        lines.append("1. Maintain current debt levels")
    
    lines.append("2. Continue timely bill payments")
    lines.append("3. Monitor credit utilization")
    
    return "\\\\n".join(lines)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{"status": "healthy", "model_loaded": ensemble_predictor is not None}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50005, debug=True)
'''.format(ensemble_model_path=ENSEMBLE_MODEL_PATH.replace('\\', '\\\\'))

# Save Flask app
flask_app_path = r"C:\Users\USER\Desktop\Personal Cashflow Risk Radar\flask_app.py"
with open(flask_app_path, 'w') as f:
    f.write(flask_app_code)

print(f"\nFlask app template saved to: {flask_app_path}")
print("\nTo run the Flask API:")
print(f"python {flask_app_path}")

print("\nAll models trained and saved successfully!")
print(f"Best ensemble R² score: {ensemble_r2:.4f}")