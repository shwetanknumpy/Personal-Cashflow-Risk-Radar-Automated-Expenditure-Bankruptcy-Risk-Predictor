# flask_app_with_llm.py - Flask API Server with LLM Explanations
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import pickle
import openai
from typing import Dict, Any

# Define the ensemble predictor class FIRST (same as in training)
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

app = Flask(__name__)

# Configuration
MODEL_DIR = r"/Users/shwetank/Desktop/kannu"
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_predictor.pkl")
FEATURES_META_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# LLM Configuration
OPENAI_API_KEY = "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"  # Replace with your actual API key
USE_LLM = True  # Set to False to disable LLM and use rule-based only

def load_model():
    """Load model with detailed error handling"""
    try:
        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            return None, f"Model file not found at: {ENSEMBLE_MODEL_PATH}"
        
        ensemble_predictor = joblib.load(ENSEMBLE_MODEL_PATH)
        
        # Verify the model has required methods
        if not hasattr(ensemble_predictor, 'predict'):
            return None, "Loaded model doesn't have predict method"
            
        print("✓ Ensemble model loaded successfully")
        return ensemble_predictor, None
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_feature_cols():
    """Load feature columns"""
    try:
        if not os.path.exists(FEATURES_META_PATH):
            return None, f"Feature file not found at: {FEATURES_META_PATH}"
        
        with open(FEATURES_META_PATH, 'r') as f:
            feature_cols = json.load(f)
        return feature_cols, None
    except Exception as e:
        return None, f"Error loading features: {str(e)}"

def initialize_openai():
    """Initialize OpenAI client"""
    try:
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
            openai.api_key = OPENAI_API_KEY
            # Test the connection with a simple request
            print("✓ OpenAI client initialized successfully")
            return True
        else:
            print("⚠️  OpenAI API key not set. LLM features disabled.")
            return False
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI: {e}")
        return False

def generate_llm_explanation(pred_score: float, features: Dict[str, Any], rule_based_explanation: str) -> str:
    """
    Generate detailed explanation using OpenAI GPT
    """
    if not USE_LLM or not openai.api_key:
        return rule_based_explanation
    
    try:
        # Extract key financial metrics
        income = features.get('INCOME', 0)
        savings = features.get('SAVINGS', 0)
        debt = features.get('DEBT', 0)
        debt_to_income = features.get('R_DEBT_INCOME', 0)
        savings_to_income = features.get('R_SAVINGS_INCOME', 0)
        
        # Calculate additional metrics
        total_expenditure = features.get('T_EXPENDITURE_12', 0)
        savings_rate = (savings / income * 100) if income > 0 else 0
        debt_utilization = (debt / (income * 10)) * 100 if income > 0 else 0  # Rough estimate
        
        prompt = f"""
        As a financial advisor, analyze this credit profile and provide a detailed, personalized explanation:

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

        Please provide:
        1. Credit score interpretation and what it means for loan eligibility
        2. Analysis of financial strengths and weaknesses
        3. Specific, actionable recommendations to improve the credit score
        4. Timeline expectations for improvement
        5. Any red flags or areas of concern

        Keep the response professional, empathetic, and focused on practical advice.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable and empathetic financial advisor specializing in credit analysis and personal finance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return f"{rule_based_explanation}\n\n[Note: Advanced AI analysis temporarily unavailable]"

def explain_rule_based(pred_score, features_dict):
    """Enhanced rule-based explanation"""
    lines = []
    
    # Interpret score
    if pred_score >= 780:
        lines.append(f"Credit Score: {pred_score:.0f} — EXCELLENT")
        lines.append("You have exceptional credit! You'll qualify for the best rates and terms.")
    elif pred_score >= 720:
        lines.append(f"Credit Score: {pred_score:.0f} — VERY GOOD")
        lines.append("Strong credit profile with access to favorable lending terms.")
    elif pred_score >= 680:
        lines.append(f"Credit Score: {pred_score:.0f} — GOOD")
        lines.append("Solid credit standing with good approval chances.")
    elif pred_score >= 620:
        lines.append(f"Credit Score: {pred_score:.0f} — FAIR")
        lines.append("Average credit score with room for improvement.")
    else:
        lines.append(f"Credit Score: {pred_score:.0f} — POOR")
        lines.append("Credit needs significant improvement for better financial opportunities.")
    
    # Key financial metrics analysis
    income = features_dict.get('INCOME', 0)
    debt = features_dict.get('DEBT', 0)
    savings = features_dict.get('SAVINGS', 0)
    debt_to_income = features_dict.get('R_DEBT_INCOME', 0)
    
    lines.append("\nFINANCIAL ANALYSIS:")
    
    if debt_to_income > 8:
        lines.append("⚠️  High debt burden - consider debt reduction strategies")
    elif debt_to_income > 5:
        lines.append("📊 Moderate debt level - manageable but watch spending")
    else:
        lines.append("✅ Healthy debt-to-income ratio")
    
    if savings > income * 0.3:
        lines.append("💰 Strong savings position")
    elif savings > income * 0.1:
        lines.append("📈 Adequate emergency fund")
    else:
        lines.append("💡 Consider building larger savings buffer")
    
    # Spending analysis
    housing_ratio = features_dict.get('R_HOUSING_INCOME', 0)
    if housing_ratio > 0.3:
        lines.append("🏠 Housing costs are high relative to income")
    
    # Recommendations
    lines.append("\nRECOMMENDATIONS:")
    if pred_score < 680:
        lines.append("1. Reduce credit card balances below 30% of limits")
        lines.append("2. Ensure all bills are paid on time")
        lines.append("3. Avoid new credit applications for 6-12 months")
    
    if debt_to_income > 5:
        lines.append("4. Focus on paying down high-interest debt first")
    
    if features_dict.get('CAT_GAMBLING', 0) == 2:  # High gambling
        lines.append("5. Consider reducing discretionary spending on entertainment")
    
    lines.append("6. Monitor credit report regularly for errors")
    lines.append("7. Maintain diverse credit mix (installment + revolving)")
    
    return "\n".join(lines)

# Load model on startup
ensemble_predictor, model_error = load_model()
feature_cols, feature_error = load_feature_cols()
openai_initialized = initialize_openai()

if model_error:
    print(f"⚠️  Model loading failed: {model_error}")
    print("💡 Run train_final.py to create models")
else:
    print("✓ Model loaded successfully")

if feature_error:
    print(f"⚠️  Feature loading failed: {feature_error}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not ensemble_predictor:
            return jsonify({
                "error": "Model not loaded", 
                "details": model_error,
                "solution": "Run train_final.py to create models first"
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = ensemble_predictor.predict(input_df)[0]
        
        # Generate rule-based explanation
        rule_based_explanation = explain_rule_based(prediction, data)
        
        # Generate LLM explanation (if enabled)
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
    """Convert numeric score to credit rating"""
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
    """Return sample input data structure"""
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
    """Check LLM configuration status"""
    return jsonify({
        "llm_enabled": USE_LLM,
        "openai_configured": openai_initialized,
        "api_key_set": bool(OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here")
    })

if __name__ == '__main__':
    print("Starting Flask server with LLM integration...")
    print("Available endpoints:")
    print("  GET  /health      - Check server status")
    print("  GET  /features    - Get required features")
    print("  GET  /sample      - Get sample input")
    print("  GET  /llm_status  - Check LLM configuration")
    print("  POST /predict     - Make prediction with AI analysis")
    
    if not ensemble_predictor:
        print("\n⚠️  WARNING: Model not loaded!")
        print("💡 Run: python train_final.py")
    
    if not openai_initialized:
        print("\n⚠️  LLM features disabled!")
        print("💡 Set OPENAI_API_KEY in the script to enable AI explanations")
    
    app.run(host='0.0.0.0', port=5005, debug=False)
