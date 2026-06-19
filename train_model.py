import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

def train_and_evaluate():
    print("Loading dataset...")
    df = pd.read_csv('credit_data.csv')
    
    # 1. Feature Engineering
    print("Performing feature engineering...")
    df['Savings_to_Income_Ratio'] = df['Savings_Balance'] / df['Annual_Income']
    df['Checking_to_Income_Ratio'] = df['Checking_Balance'] / df['Annual_Income']
    df['CC_Utilization_Risk'] = (df['Credit_Card_Utilization'] > 0.7).astype(int)
    df['DTI_High'] = (df['Debt_to_Income_Ratio'] > 0.4).astype(int)
    
    # Target and features
    # Exclude ID, Credit_Score (since it's a direct proxy of the label, we want to predict based on financial parameters)
    # and Credit_Status (target)
    target_col = 'Credit_Status'
    exclude_cols = ['Applicant_ID', 'Credit_Score', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define features categories for ColumnTransformer
    categorical_cols = ['Education_Level']
    # Owns_Home, CC_Utilization_Risk, DTI_High are already binary (0/1) but can be treated as numerical/passthrough
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    results = {}
    best_f1 = -1
    best_model_name = None
    best_pipeline = None
    
    # Create static/images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Plot setup for ROC curve
    plt.figure(figsize=(8, 6))
    
    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'auc': round(auc, 4)
        }
        
        print(f"--- {name} Results ---")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        # Save best model based on F1 score (or AUC)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline
            
        # Add to ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        
    # Finalize ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/roc_curve.png', dpi=300)
    plt.close()
    
    print(f"\nBest Model selected: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # 2. Confusion Matrix for Best Model
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Bad Credit (0)', 'Good Credit (1)'])
    plt.yticks(tick_marks, ['Bad Credit (0)', 'Good Credit (1)'])
    
    # Text annotations in Confusion Matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 3. Feature Importance for Best Model (if Random Forest or Decision Tree)
    classifier = best_pipeline.named_steps['classifier']
    preprocessor_transformer = best_pipeline.named_steps['preprocessor']
    
    # Get feature names after preprocessing
    cat_encoder = preprocessor_transformer.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + encoded_cat_features
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 10 features
        top_n = min(12, len(all_feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(8, 6))
        plt.title(f'Top {top_n} Feature Importances ({best_model_name})')
        plt.barh(range(top_n), importances[top_indices][::-1], align='center', color='#4f46e5')
        plt.yticks(range(top_n), [all_feature_names[i] for i in top_indices][::-1])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('static/images/feature_importance.png', dpi=300)
        plt.close()
    elif hasattr(classifier, 'coef_'):
        # LogReg coefficients
        coefs = classifier.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]
        top_n = min(12, len(all_feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(8, 6))
        plt.title(f'Top {top_n} Feature Coefficients ({best_model_name})')
        plt.barh(range(top_n), coefs[top_indices][::-1], align='center', color='#4f46e5')
        plt.yticks(range(top_n), [all_feature_names[i] for i in top_indices][::-1])
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig('static/images/feature_importance.png', dpi=300)
        plt.close()
        
    # Save the model pipeline and configuration
    model_data = {
        'model_name': best_model_name,
        'pipeline': best_pipeline,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'metrics': results
    }
    
    with open('credit_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model and metrics successfully saved to 'credit_model.pkl'.")

if __name__ == "__main__":
    train_and_evaluate()
