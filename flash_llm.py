from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import pickle
import openai
from typing import Dict, Any

class EnsemblePredictor:
    def __init__(self, nn_model, rf_model, ridge_model, scaler, feature_cols, weights=[0.3, 0.4, 0.3]):
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.ridge_model = ridge_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.weights = weights
        
    def predict(self, input_df):
        missing = [c for c in self.feature_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X_in = input_df[self.feature_cols].values
        X_in_scaled = self.scaler.transform(X_in)
        nn_pred = self.nn_model.predict(X_in_scaled)
        rf_pred = self.rf_model.predict(X_in_scaled)
        ridge_pred = self.ridge_model.predict(X_in_scaled)
        ensemble_pred = (self.weights[0] * nn_pred + self.weights[1] * rf_pred + self.weights[2] * ridge_pred)
        ensemble_pred = np.clip(ensemble_pred, 300.0, 850.0)
        return ensemble_pred

app = Flask(__name__)

MODEL_DIR = r"/Users/shwetank/Desktop/kannu"
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_predictor.pkl")
FEATURES_META_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

OPENAI_API_KEY = "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"
USE_LLM = True

def load_model():
    try:
        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            return None, f"Model file not found at: {ENSEMBLE_MODEL_PATH}"
        ensemble_predictor = joblib.load(ENSEMBLE_MODEL_PATH)
        if not hasattr(ensemble_predictor, 'predict'):
            return None, "Loaded model doesn't have predict method"
        print("✓ Ensemble model loaded successfully")
        return ensemble_predictor, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_feature_cols():
    try:
        if not os.path.exists(FEATURES_META_PATH):
            return None, f"Feature file not found at: {FEATURES_META_PATH}"
        with open(FEATURES_META_PATH, 'r') as f:
            feature_cols = json.load(f)
        return feature_cols, None
    except Exception as e:
        return None, f"Error loading features: {str(e)}"

def initialize_openai():
    try:
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
            openai.api_key = OPENAI_API_KEY
            print("✓ OpenAI client initialized successfully")
            return True
        else:
            print("⚠️  OpenAI API key not set. LLM features disabled.")
            return False
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI: {e}")
        return False

def generate_llm_explanation(pred_score: float, features: Dict[str, Any], rule_based_explanation: str) -> str:
    if not USE_LLM or not openai.api_key:
        return rule_based_explanation
    try:
        income = features.get('INCOME', 0)
        savings = features.get('SAVINGS', 0)
        debt = features.get('DEBT', 0)
        debt_to_income = features.get('R_DEBT_INCOME', 0)
        savings_to_income = features.get('R_SAVINGS_INCOME', 0)
        total_expenditure = features.get('T_EXPENDITURE_12', 0)
        savings_rate = (savings / income * 100) if income > 0 else 0
        debt_utilization = (debt / (income * 10)) * 100 if income > 0 else 0
        
        prompt = f"""
        As a financial advisor, analyze this credit profile and provide a detailed explanation:

        CREDIT SCORE: {pred_score:.0f}
        FINANCIAL SNAPSHOT:
        - Annual Income: ${income:,.0f}
        - Savings: ${savings:,.0f}
        - Total Debt: ${debt:,.0f}
        - Debt-to-Income Ratio: {debt_to_income:.1f}x
        - Savings-to-Income Ratio: {savings_to_income:.1f}x
        - Annual Expenditure: ${total_expenditure:,.0f}
        - Savings Rate: {savings_rate:.1f}%
        - Debt Utilization: {debt_utilization:.1f}%

        SPENDING PATTERNS:
        - Housing: ${features.get('T_HOUSING_12', 0):,.0f}
        - Groceries: ${features.get('T_GROCERIES_12', 0):,.0f}
        - Entertainment: ${features.get('T_ENTERTAINMENT_12', 0):,.0f}
        - Travel: ${features.get('T_TRAVEL_12', 0):,.0f}
        - Utilities: ${features.get('T_UTILITIES_12', 0):,.0f}

        FINANCIAL BEHAVIORS:
        - Gambling Activity: {['None', 'Low', 'High'][features.get('CAT_GAMBLING', 0)]}
        - Has Credit Card: {'Yes' if features.get('CAT_CREDIT_CARD', 0) else 'No'}
        - Has Mortgage: {'Yes' if features.get('CAT_MORTGAGE', 0) else 'No'}
        - Has Savings Account: {'Yes' if features.get('CAT_SAVINGS_ACCOUNT', 0) else 'No'}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return f"{rule_based_explanation}\n\n[Note: AI analysis temporarily unavailable]"

def explain_rule_based(pred_score, features_dict):
    lines = []
    if pred_score >= 780:
        lines.append(f"Credit Score: {pred_score:.0f} — EXCELLENT")
    elif pred_score >= 720:
        lines.append(f"Credit Score: {pred_score:.0f} — VERY GOOD")
    elif pred_score >= 680:
        lines.append(f"Credit Score: {pred_score:.0f} — GOOD")
    elif pred_score >= 620:
        lines.append(f"Credit Score: {pred_score:.0f} — FAIR")
    else:
        lines.append(f"Credit Score: {pred_score:.0f} — POOR")

    income = features_dict.get('INCOME', 0)
    debt = features_dict.get('DEBT', 0)
    savings = features_dict.get('SAVINGS', 0)
    debt_to_income = features_dict.get('R_DEBT_INCOME', 0)
    lines.append("\nFINANCIAL ANALYSIS:")
    if debt_to_income > 8:
        lines.append("⚠️ High debt burden - consider debt reduction")
    elif debt_to_income > 5:
        lines.append("📊 Moderate debt level")
    else:
        lines.append("✅ Healthy debt ratio")

    if savings > income * 0.3:
        lines.append("💰 Strong savings position")
    elif savings > income * 0.1:
        lines.append("📈 Adequate emergency fund")
    else:
        lines.append("💡 Increase savings buffer")

    lines.append("\nRECOMMENDATIONS:")
    if pred_score < 680:
        lines.append("1. Reduce credit card balances below 30% of limits")
        lines.append("2. Pay bills on time")
        lines.append("3. Avoid new credit applications")
    if debt_to_income > 5:
        lines.append("4. Pay down high-interest debt")
    if features_dict.get('CAT_GAMBLING', 0) == 2:
        lines.append("5. Reduce discretionary spending")
    lines.append("6. Monitor credit report regularly")
    lines.append("7. Maintain diverse credit mix")
    return "\n".join(lines)

ensemble_predictor, model_error = load_model()
feature_cols, feature_error = load_feature_cols()
openai_initialized = initialize_openai()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not ensemble_predictor:
            return jsonify({
                "error": "Model not loaded", 
                "details": model_error,
                "solution": "Run train_final.py to create models"
            }), 500
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        input_df = pd.DataFrame([data])
        prediction = ensemble_predictor.predict(input_df)[0]
        rule_based_explanation = explain_rule_based(prediction, data)
        llm_explanation = generate_llm_explanation(prediction, data, rule_based_explanation)
        response_data = {
            "predicted_credit_score": float(prediction),
            "credit_rating": get_credit_rating(prediction),
            "rule_based_explanation": rule_based_explanation,
            "ai_analysis": llm_explanation,
            "llm_enabled": USE_LLM and openai_initialized,
            "status": "success"
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_credit_rating(score: float) -> str:
    if score >= 780:
        return "Excellent"
    elif score >= 720:
        return "Very Good"
    elif score >= 680:
        return "Good"
    elif score >= 620:
        return "Fair"
    else:
        return "Poor"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if ensemble_predictor else "unhealthy",
        "model_loaded": ensemble_predictor is not None,
        "model_error": model_error if not ensemble_predictor else None,
        "features_loaded": feature_cols is not None,
        "llm_enabled": USE_LLM and openai_initialized,
        "openai_configured": openai_initialized
    })

@app.route('/features', methods=['GET'])
def features():
    if feature_cols:
        return jsonify({
            "required_features": feature_cols,
            "total_features": len(feature_cols)
        })
    else:
        return jsonify({
            "error": "Features not loaded",
            "details": feature_error
        }), 500

@app.route('/sample', methods=['GET'])
def sample():
    sample_data = {
        "INCOME": 50000,
        "SAVINGS": 25000,
        "DEBT": 150000,
        "R_SAVINGS_INCOME": 0.5,
        "R_DEBT_INCOME": 3.0,
        "R_DEBT_SAVINGS": 6.0,
        "T_CLOTHING_12": 1200,
        "T_CLOTHING_6": 600,
        "R_CLOTHING": 0.5,
        "R_CLOTHING_INCOME": 0.024,
        "CAT_GAMBLING": 1
    }
    return jsonify({"sample_input": sample_data})

@app.route('/llm_status', methods=['GET'])
def llm_status():
    return jsonify({
        "llm_enabled": USE_LLM,
        "openai_configured": openai_initialized,
        "api_key_set": bool(OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here")
    })

if __name__ == '__main__':
    print("Starting Flask server with LLM integration...")
    app.run(host='0.0.0.0', port=5005, debug=False)
