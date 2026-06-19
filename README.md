# CrediShield AI: Credit Scoring & Risk Assessment Dashboard

An interactive, premium machine learning web application that predicts an individual's creditworthiness and evaluates default risk using past financial and credit history.

---

## 🌟 Key Features

1. **Synthetic Financial Dataset Generator**: Creates realistic financial records for 5,000 applicants with synthetic demographics, assets, debt profiles, CC utilization, payment delays, and computed FICO scores.
2. **Machine Learning Pipeline**:
   - Compares **Logistic Regression**, **Decision Trees**, and **Random Forests**.
   - Standardizes numerical variables and encodes categorical features using a robust preprocessing pipeline.
   - Saves the optimal model, scaler, encoder, and metadata as a unified pipeline artifact (`credit_model.pkl`).
3. **Model Evaluation Charts**: Generates and renders static assets for model diagnostic review (ROC Curves, Confusion Matrix, Feature Coefficients/Importances).
4. **Interactive Risk Dashboard**:
   - **Applicant Risk Diagnostics**: Instantly computes applicant credit scores, risk levels, and loan decisions (Approved vs. Rejected).
   - **Dynamic Risk Gauge**: Animated circular risk meter representing the applicant's credit score.
   - **Explainable Decisions**: Highlights key positive drivers and critical warning flags influencing the model decision.
   - **Model Analytics Hub**: Real-time performance tables and ML visualization assets directly in the UI.

---

## 📂 Project Structure

```text
codealpha task 1/
│
├── static/
│   ├── images/                 # Generated evaluation plots
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── roc_curve.png
│   │
│   ├── app.js                  # Frontend form handling and SVG gauge rendering
│   └── style.css               # Premium dark glassmorphism styling
│
├── templates/
│   └── index.html              # HTML5 responsive structure and dashboard views
│
├── app.py                      # Flask web server and prediction routes
├── generate_dataset.py         # Custom dataset generator (5,000 records)
├── train_model.py              # ML training, evaluation, and pipeline pickling script
├── requirements.txt            # Project python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
Create the synthetic credit history dataset:
```bash
python generate_dataset.py
```
*Creates `credit_data.csv` with a representative customer mix (Good vs. Bad credit status).*

### 3. Train and Save the Model
Execute the training script to test multiple classification algorithms, select the best model, and generate visual charts:
```bash
python train_model.py
```
*Trains Logistic Regression, Decision Trees, and Random Forests. Saves the best pipeline (e.g., Logistic Regression) to `credit_model.pkl` and outputs graphs in `static/images/`.*

### 4. Run the Web Dashboard
Launch the Flask application locally:
```bash
python app.py
```
Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 📊 Model Benchmarking Details

During the training pipeline run, the models are evaluated using key performance metrics:

- **ROC-AUC**: Shows how well the model distinguishes between approved and default-prone classes.
- **Precision**: Focuses on minimizing false positives (approving a high-risk applicant).
- **Recall**: Focuses on identifying all high-risk applicants.
- **F1-Score**: Harmonic mean of Precision and Recall.

*The model with the highest F1-Score is chosen as the deployment engine.*

---

## 🛡️ License
This project is open-source and developed as part of task completion.
