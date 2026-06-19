import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_credit_data(num_samples=5000):
    print(f"Generating synthetic financial dataset with {num_samples} records...")
    
    # 1. Demographic data
    age = np.random.normal(loc=39, scale=12, size=num_samples).astype(int)
    age = np.clip(age, 18, 75)
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, size=num_samples, p=[0.35, 0.45, 0.15, 0.05])
    
    # 2. Employment and Income
    # Employment years correlates loosely with age
    employment_years = []
    for a in age:
        max_work_years = max(0, a - 18)
        emp_yr = np.random.exponential(scale=5)
        emp_yr = min(emp_yr, max_work_years)
        employment_years.append(round(emp_yr, 1))
    employment_years = np.array(employment_years)
    
    # Income (log-normal, to represent wealth skewness)
    annual_income = np.random.lognormal(mean=10.9, sigma=0.5, size=num_samples)
    annual_income = np.clip(annual_income, 15000, 250000).round(2)
    
    # Home ownership correlates with age and income
    own_home_prob = []
    for a, inc in zip(age, annual_income):
        prob = 0.1 + (a / 100) * 0.4 + (inc / 250000) * 0.4
        prob = np.clip(prob, 0.05, 0.9)
        own_home_prob.append(prob)
    owns_home = np.array([np.random.choice([1, 0], p=[p, 1-p]) for p in own_home_prob])
    
    # 3. Bank Relationship
    # Savings and checking balances correlate with income
    savings_balance = (annual_income * np.random.uniform(0.01, 0.25, num_samples) + np.random.uniform(0, 5000, num_samples)).round(2)
    # Give some people 0 savings
    zero_savings_mask = np.random.choice([True, False], size=num_samples, p=[0.15, 0.85])
    savings_balance[zero_savings_mask] = 0.0
    
    checking_balance = (annual_income * np.random.uniform(0.005, 0.1, num_samples) + np.random.uniform(0, 2000, num_samples)).round(2)
    # Give some people 0 checking
    zero_checking_mask = np.random.choice([True, False], size=num_samples, p=[0.08, 0.92])
    checking_balance[zero_checking_mask] = 0.0
    
    # 4. Debt and Loan attributes
    existing_loans = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.45, 0.35, 0.13, 0.05, 0.02])
    
    credit_card_utilization = np.random.beta(a=2, b=5, size=num_samples) # skewed towards lower utilization, but some high
    credit_card_utilization = np.clip(credit_card_utilization, 0.0, 1.0).round(3)
    
    loan_amount = np.random.normal(loc=25000, scale=18000, size=num_samples)
    loan_amount = np.clip(loan_amount, 1000, 120000).round(2)
    
    loan_duration_months = np.random.choice([12, 24, 36, 48, 60], size=num_samples, p=[0.1, 0.2, 0.4, 0.1, 0.2])
    
    # Calculate DTI (Debt-to-Income) ratio
    # Monthly loan payment estimate (simple linear split + 5% interest estimate)
    est_monthly_payment = (loan_amount / loan_duration_months) * 1.05
    monthly_income = annual_income / 12
    dti_ratio = (est_monthly_payment / monthly_income).round(3)
    
    # Payment delays (Poisson distribution)
    # Delay probability higher if DTI is high or CC utilization is high, or savings is low
    payment_delays = []
    for dti, cc_util, savings in zip(dti_ratio, credit_card_utilization, savings_balance):
        lam = 0.2 + (dti * 2.0) + (cc_util * 1.5)
        if savings < 2000:
            lam += 0.5
        delays = np.random.poisson(lam)
        payment_delays.append(min(delays, 12)) # cap at 12 late payments
    payment_delays = np.array(payment_delays)
    
    # 5. Calculate Synthetic Credit Score to drive target label
    # FICO range: 300 to 850
    base_score = 600
    
    # Score component calculations
    age_component = (age - 18) * 1.2
    income_component = np.log10(annual_income / 10000 + 1) * 35
    savings_component = np.log10(savings_balance / 1000 + 1) * 30
    checking_component = np.log10(checking_balance / 1000 + 1) * 15
    dti_component = - (dti_ratio * 120)
    cc_component = - (credit_card_utilization * 150)
    delay_component = - (payment_delays * 45)
    emp_component = employment_years * 2.5
    home_component = owns_home * 35
    loans_component = - (existing_loans * 15)
    
    computed_scores = (
        base_score + 
        age_component + 
        income_component + 
        savings_component + 
        checking_component + 
        dti_component + 
        cc_component + 
        delay_component + 
        emp_component + 
        home_component + 
        loans_component
    )
    
    # Clip credit scores between 300 and 850
    credit_scores = np.clip(computed_scores, 300, 850).astype(int)
    
    # 6. Map Credit Score to target 'Credit_Status' (1 = Good, 0 = Bad) with realistic probability
    # FICO ranges:
    # >= 720: Excellent/Good (very high approval probability)
    # 660-719: Fair (moderate approval probability)
    # 600-659: Subprime (lower approval probability)
    # < 600: Poor (very low approval probability)
    credit_status = []
    for score in credit_scores:
        if score >= 720:
            prob = 0.96
        elif score >= 660:
            prob = 0.85
        elif score >= 600:
            prob = 0.55
        elif score >= 550:
            prob = 0.25
        else:
            prob = 0.05
        
        status = np.random.choice([1, 0], p=[prob, 1-prob])
        credit_status.append(status)
        
    credit_status = np.array(credit_status)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Applicant_ID': [f"ID_{1001 + i}" for i in range(num_samples)],
        'Age': age,
        'Education_Level': education,
        'Employment_Duration_Years': employment_years,
        'Annual_Income': annual_income,
        'Owns_Home': owns_home,
        'Savings_Balance': savings_balance,
        'Checking_Balance': checking_balance,
        'Existing_Loans_Count': existing_loans,
        'Credit_Card_Utilization': credit_card_utilization,
        'Loan_Amount': loan_amount,
        'Loan_Duration_Months': loan_duration_months,
        'Debt_to_Income_Ratio': dti_ratio,
        'Payment_History_Delay': payment_delays,
        'Credit_Score': credit_scores,
        'Credit_Status': credit_status
    })
    
    # Save to CSV
    output_path = 'credit_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to '{output_path}'.")
    print(f"Class distribution: {df['Credit_Status'].value_counts(normalize=True).to_dict()}")
    print(f"Average Credit Score: {df['Credit_Score'].mean():.1f}")
    
    return df

if __name__ == "__main__":
    generate_credit_data()
