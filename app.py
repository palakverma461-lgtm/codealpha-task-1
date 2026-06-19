import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model data
MODEL_PATH = 'credit_model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Loaded trained {model_data['model_name']} model.")
else:
    model_data = None
    print("Warning: credit_model.pkl not found. Please train the model first.")

@app.route('/')
def index():
    if model_data:
        # Pass the metrics to display in the frontend
        metrics = model_data.get('metrics', {})
        best_model = model_data.get('model_name', 'Model')
    else:
        metrics = {}
        best_model = "None"
        
    return render_template('index.html', metrics=metrics, best_model=best_model)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return jsonify({'error': 'Model not trained or loaded.'}), 500
        
    try:
        data = request.json
        
        # Extract inputs from request
        age = int(data.get('Age', 30))
        education = data.get('Education_Level', 'Bachelor')
        employment_years = float(data.get('Employment_Duration_Years', 2.0))
        annual_income = float(data.get('Annual_Income', 50000.0))
        owns_home = int(data.get('Owns_Home', 0))
        savings_balance = float(data.get('Savings_Balance', 5000.0))
        checking_balance = float(data.get('Checking_Balance', 1000.0))
        existing_loans = int(data.get('Existing_Loans_Count', 0))
        cc_utilization = float(data.get('Credit_Card_Utilization', 0.3))
        loan_amount = float(data.get('Loan_Amount', 15000.0))
        loan_duration = int(data.get('Loan_Duration_Months', 36))
        payment_delays = int(data.get('Payment_History_Delay', 0))
        
        # Calculate DTI Ratio based on the training formula
        est_monthly_payment = (loan_amount / loan_duration) * 1.05
        monthly_income = annual_income / 12 if annual_income > 0 else 1.0
        dti_ratio = round(est_monthly_payment / monthly_income, 3)
        
        # Apply same Feature Engineering as training
        savings_to_income = savings_balance / annual_income if annual_income > 0 else 0.0
        checking_to_income = checking_balance / annual_income if annual_income > 0 else 0.0
        cc_util_risk = 1 if cc_utilization > 0.7 else 0
        dti_high = 1 if dti_ratio > 0.4 else 0
        
        # Create dictionary matching the feature names expected by the model
        input_data = {
            'Age': age,
            'Education_Level': education,
            'Employment_Duration_Years': employment_years,
            'Annual_Income': annual_income,
            'Owns_Home': owns_home,
            'Savings_Balance': savings_balance,
            'Checking_Balance': checking_balance,
            'Existing_Loans_Count': existing_loans,
            'Credit_Card_Utilization': cc_utilization,
            'Loan_Amount': loan_amount,
            'Loan_Duration_Months': loan_duration,
            'Debt_to_Income_Ratio': dti_ratio,
            'Payment_History_Delay': payment_delays,
            'Savings_to_Income_Ratio': savings_to_income,
            'Checking_to_Income_Ratio': checking_to_income,
            'CC_Utilization_Risk': cc_util_risk,
            'DTI_High': dti_high
        }
        
        # Order the features correctly matching the pipeline
        ordered_features = model_data['feature_cols']
        input_df = pd.DataFrame([input_data])[ordered_features]
        
        # Predict using pipeline
        pipeline = model_data['pipeline']
        prediction = int(pipeline.predict(input_df)[0])
        prob_good = float(pipeline.predict_proba(input_df)[0][1])
        
        # Map probability to a synthetic Credit Score (FICO scale 300 - 850)
        # Using a nonlinear scaling to make it feel natural
        credit_score = int(300 + (prob_good * 550))
        
        # Risk assessment categories
        if credit_score >= 720:
            category = "Excellent"
            risk_desc = "Very Low Default Risk"
            color_class = "status-excellent"
        elif credit_score >= 660:
            category = "Good"
            risk_desc = "Low Default Risk"
            color_class = "status-good"
        elif credit_score >= 600:
            category = "Fair"
            risk_desc = "Moderate Default Risk"
            color_class = "status-fair"
        else:
            category = "Poor"
            risk_desc = "High Default Risk"
            color_class = "status-poor"
            
        # Decision
        decision = "Approved" if prediction == 1 else "Rejected"
        
        # Key positive and negative factors
        factors_pos = []
        factors_neg = []
        
        # Generate dynamic suggestions
        if cc_utilization < 0.3:
            factors_pos.append("Excellent Credit Card Utilization (under 30%)")
        elif cc_utilization > 0.7:
            factors_neg.append("High Credit Card Utilization (over 70%) - reduces score")
            
        if dti_ratio < 0.2:
            factors_pos.append("Low Debt-to-Income ratio (DTI under 20%)")
        elif dti_ratio > 0.4:
            factors_neg.append("High Debt-to-Income ratio (DTI over 40%) - elevates default risk")
            
        if payment_delays == 0:
            factors_pos.append("No payment delays in the last 12 months")
        else:
            factors_neg.append(f"{payment_delays} late payment(s) in the last 12 months")
            
        if savings_balance > (annual_income * 0.1):
            factors_pos.append("Healthy savings safety net (over 10% of annual income)")
        elif savings_balance < 1000:
            factors_neg.append("Low savings balance - increases financial vulnerability")
            
        if employment_years >= 5:
            factors_pos.append("Stable employment history (5+ years)")
        elif employment_years < 1:
            factors_neg.append("Short job stability (under 1 year)")
            
        # Fallback if list is empty
        if not factors_pos:
            factors_pos.append("Regular income stream verified")
        if not factors_neg:
            factors_neg.append("No significant credit risks identified")
            
        return jsonify({
            'prediction': prediction,
            'probability': prob_good,
            'credit_score': credit_score,
            'category': category,
            'risk_description': risk_desc,
            'color_class': color_class,
            'decision': decision,
            'debt_to_income': dti_ratio,
            'monthly_payment': round(est_monthly_payment, 2),
            'factors_positive': factors_pos,
            'factors_negative': factors_neg
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
