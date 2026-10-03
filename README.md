# 💳 Credit Risk Prediction using Machine Learning

An end-to-end **Credit Risk Prediction** project that uses machine learning classification models to predict whether a loan applicant is likely to be **high risk or low risk**.

The project includes **EDA, data preprocessing, multiple ML models, XGBoost, SHAP-based explainability, an interactive Streamlit application, and cloud deployment**.

## 🚀 Live Demo

👉 **[Try the Live Streamlit App](https://credit-risk-ml-project-5ouz32jtmnxhjckag8xuza.streamlit.app/)**

---

## 📌 Project Overview

Credit risk assessment is an important task in the lending industry. The goal of this project is to build a machine learning system that analyzes applicant and loan information and predicts the associated credit risk.

The application allows users to:

* Enter applicant details
* Select a machine learning model
* Predict credit risk
* View risk probability
* Understand the prediction using SHAP
* View feature contributions
* Maintain prediction history
* Download prediction history as a CSV file

---

## 🧠 Machine Learning Models

The application provides three classification models:

| Model                | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| 🌲 Random Forest     | Ensemble tree-based classifier                       |
| 🚀 Gradient Boosting | Sequential boosting-based classifier                 |
| ⚡ XGBoost            | Gradient boosting framework used for the final model |

The application allows users to select the model they want to use for prediction.

---

## 📊 Dataset

The project uses a credit risk dataset containing applicant and loan-related information.

### Input Features

* `person_age` — Applicant age
* `person_income` — Applicant annual income
* `person_home_ownership` — Home ownership status
* `person_emp_length` — Employment length
* `loan_intent` — Purpose of the loan
* `loan_grade` — Loan grade
* `loan_amnt` — Loan amount
* `loan_int_rate` — Loan interest rate
* `loan_percent_income` — Loan amount as a percentage of income
* `cb_person_default_on_file` — Previous default indicator
* `cb_person_cred_hist_length` — Credit history length

### Target

`loan_status`

The model uses the target labels from the dataset to classify credit risk.

---

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis such as:

* Dataset structure and information
* Missing value analysis
* Target distribution
* Numerical feature distributions
* Outlier analysis
* Correlation analysis
* Feature relationships
* Pairwise analysis
* Class imbalance analysis

The complete analysis is available in:

```text
CREDIT_RISK.ipynb
```

---

## ⚙️ Machine Learning Pipeline

The project follows a structured ML workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
SHAP Explainability
   ↓
Streamlit Application
   ↓
Deployment
```

Categorical and numerical features are handled through preprocessing before being passed to the machine learning models.

---

## 🔬 Model Explainability with SHAP

The application uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For every prediction, the application provides:

* Top contributing features
* SHAP contribution values
* Feature contribution bar chart
* SHAP waterfall plot
* Explanation of positive and negative contributions

This helps users understand **why the model produced a particular prediction** rather than only seeing the final result.

---

## 💻 Streamlit Application

The Streamlit application provides an interactive interface where users can enter applicant information.

### Application Features

### 🤖 Model Selection

Users can select:

```text
Random Forest
Gradient Boosting
XGBoost
```

### 📝 Applicant Input

Users can enter:

* Age
* Income
* Home ownership
* Employment length
* Loan intent
* Loan grade
* Loan amount
* Interest rate
* Loan percentage of income
* Previous default
* Credit history length

### 🎯 Prediction

The application displays:

* Predicted credit risk
* Risk probability
* Probability interpretation

### 🔍 Explainability

SHAP provides detailed feature-level explanations for the prediction.

### 📜 Prediction History

Previous predictions are stored during the current application session and can be downloaded as a CSV file.

---

## 📂 Project Structure

```text
Credit-Risk-ML-Project/
│
├── app.py
├── CREDIT_RISK.ipynb
├── credit_risk_dataset.csv
│
├── credit_risk_preprocessor.pkl
├── credit_risk_rf_model.pkl
├── credit_risk_gb_model.pkl
├── credit_risk_xgb_model.json
├── credit_risk_xgb_pipeline.pkl
│
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Web Application

* Streamlit

### Model Persistence

* Joblib
* XGBoost JSON model format

### Development Tools

* Jupyter Notebook
* Git
* GitHub
* VS Code

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/BhanuPrakashReddy-B/Credit-Risk-ML-Project.git
```

Navigate to the project:

```bash
cd Credit-Risk-ML-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

👉 **[Credit Risk Prediction App](https://credit-risk-ml-project-5ouz32jtmnxhjckag8xuza.streamlit.app/)**

---

## 📈 Future Improvements

Possible future improvements include:

* Hyperparameter optimization
* Model performance comparison dashboard
* ROC-AUC and Precision-Recall visualization
* Probability calibration
* Custom classification threshold
* Additional explainability features
* Improved UI/UX
* Model monitoring
* Automated model retraining

---

## 👨‍💻 Author

### Bhavanam Bhanu Prakash Reddy

B.Tech CSE — AI & Data Science

GitHub: **[BhanuPrakashReddy-B](https://github.com/BhanuPrakashReddy-B)**

---

## ⭐ If You Like This Project

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.
