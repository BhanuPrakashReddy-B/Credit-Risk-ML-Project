import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)



# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    "credit_risk_preprocessor.pkl"
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

rf_model = joblib.load(
    "credit_risk_rf_model.pkl"
)


# ============================================================
# LOAD GRADIENT BOOSTING
# ============================================================

gb_model = joblib.load(
    "credit_risk_gb_model.pkl"
)


# ============================================================
# LOAD XGBOOST
# ============================================================

xgb_model = XGBClassifier()

xgb_model.load_model(
    "credit_risk_xgb_model.json"
)





# ============================================================
# PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Risk Prediction")

st.write(
    "Enter the applicant details to predict credit risk."
)


# ============================================================
# MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "🤖 Select Model",
    [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    index=2
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        f"🤖 Selected Model\n\n{selected_model}"
    )

with col2:
    st.info(
        "🎯 Task\n\nClassification"
    )

with col3:
    st.info(
        "🛡️ Output\n\nCredit Risk"
    )


# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    person_age = st.number_input(
        "Person Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Person Income",
        min_value=1,
        value=50000,
        step=1000
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=1,
        value=10000,
        step=500
    )

    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        [
            "Y",
            "N"
        ]
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0,
        max_value=50,
        value=5
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "person_age": [person_age],

    "person_income": [person_income],

    "person_home_ownership": [
        person_home_ownership
    ],

    "person_emp_length": [
        person_emp_length
    ],

    "loan_intent": [
        loan_intent
    ],

    "loan_grade": [
        loan_grade
    ],

    "loan_amnt": [
        loan_amnt
    ],

    "loan_int_rate": [
        loan_int_rate
    ],

    "loan_percent_income": [
        loan_percent_income
    ],

    "cb_person_default_on_file": [
        cb_person_default_on_file
    ],

    "cb_person_cred_hist_length": [
        cb_person_cred_hist_length
    ]
})


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    predict_button = st.button(
        "🔍 Predict Credit Risk",
        use_container_width=True
    )


with col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )


# ============================================================
# RESET
# ============================================================

if reset_button:

    st.session_state.prediction_history = []

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # SELECT MODEL
    # ========================================================

    if selected_model == "Random Forest":

        model = rf_model

        model_preprocessor = model.named_steps["preprocessor"]

        classifier = model.named_steps["classifier"]

        input_processed = model_preprocessor.transform(
            input_data
        )


    elif selected_model == "Gradient Boosting":

        model = gb_model

        model_preprocessor = model.named_steps["preprocessor"]

        classifier = model.named_steps["classifier"]

        input_processed = model_preprocessor.transform(
            input_data
        )


    else:

        model = xgb_model

        model_preprocessor = preprocessor

        classifier = xgb_model

        input_processed = model_preprocessor.transform(
            input_data
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = classifier.predict(
        input_processed
    )[0]


    probability = classifier.predict_proba(
        input_processed
    )[0][1]


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = model_preprocessor.get_feature_names_out()

    feature_names = [
        name.split("__")[-1].replace("_", " ")
        for name in feature_names
    ]

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader("🔍 Prediction Explanation")


    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier
    )


    # --------------------------------------------------------
    # CALCULATE SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer(
        input_processed
    )


    # --------------------------------------------------------
    # GET SHAP VALUES SAFELY
    # --------------------------------------------------------

    shap_contributions = shap_values.values


    # Remove unnecessary dimensions
    shap_contributions = shap_contributions.squeeze()


    # If SHAP returns values for multiple classes,
    # use the High Risk (class 1) contribution
    if shap_contributions.ndim > 1:

        shap_contributions = shap_contributions[:, 1]


    # Make sure it is 1-dimensional
    shap_contributions = shap_contributions.ravel()


    # --------------------------------------------------------
    # GET INPUT VALUES SAFELY
    # --------------------------------------------------------

    input_values = input_processed[0]

    # Convert sparse matrix / array to 1-D
    if hasattr(input_values, "toarray"):
        input_values = input_values.toarray().ravel()
    else:
        input_values = input_values.ravel()


    # --------------------------------------------------------
    # CHECK FEATURE COUNT
    # --------------------------------------------------------

    if len(feature_names) != len(shap_contributions):

        st.error(
            f"SHAP feature mismatch: "
            f"{len(feature_names)} feature names but "
            f"{len(shap_contributions)} SHAP values."
        )

        st.stop()


    # --------------------------------------------------------
    # SHAP DATAFRAME
    # --------------------------------------------------------

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "Value": input_values,

        "SHAP Contribution": shap_contributions

    })


    # --------------------------------------------------------
    # ABSOLUTE SHAP
    # --------------------------------------------------------

    shap_df["Absolute SHAP"] = (
        shap_df["SHAP Contribution"].abs()
    )


    # --------------------------------------------------------
    # SORT FEATURES
    # --------------------------------------------------------

    shap_df = shap_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )


    # ========================================================
    # TOP SHAP FEATURES
    # ========================================================

    st.write(
        "The following features had the strongest "
        "contribution to this prediction:"
    )


    top_shap = shap_df.head(10)


    st.dataframe(
        top_shap[
            [
                "Feature",
                "Value",
                "SHAP Contribution"
            ]
        ],
        use_container_width=True
    )


    # ========================================================
    # SHAP BAR CHART
    # ========================================================

    st.subheader("📊 Feature Contributions")


    chart_data = top_shap[
        [
            "Feature",
            "SHAP Contribution"
        ]
    ].set_index("Feature")


    st.bar_chart(
        chart_data
    )

    # ========================================================
    # SHAP WATERFALL
    # ========================================================

    st.subheader(
        "📈 Detailed Prediction Explanation"
    )

    # Get SHAP explanation for the first applicant
    # and High Risk class (class 1)

    if len(shap_values.shape) == 3:

        waterfall_explanation = shap_values[0, :, 1]

    elif len(shap_values.shape) == 2:

        waterfall_explanation = shap_values[0]

    else:

        waterfall_explanation = shap_values


    waterfall = shap.plots.waterfall(
        waterfall_explanation,
        max_display=10,
        show=False
    )


    st.pyplot(
        waterfall.figure
    )


    plt.close(
        waterfall.figure
    )
    # ========================================================
    # SHAP INTERPRETATION GUIDE
    # ========================================================

    with st.expander(
        "ℹ️ How to interpret SHAP"
    ):

        st.write(
            """
            SHAP values explain how individual features
            contributed to the model's prediction.
            """
        )

        st.write(
            """
            🔴 Positive SHAP contribution:
            pushes the prediction toward the High Risk class.
            """
        )

        st.write(
            """
            🔵 Negative SHAP contribution:
            pushes the prediction toward the Low Risk class.
            """
        )

        st.write(
            """
            The larger the absolute SHAP value,
            the stronger the feature's contribution
            to this prediction.
            """
        )

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ High Credit Risk"
        )

    else:

        st.success(
            "✅ Low Credit Risk"
        )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    st.metric(
        "Risk Probability",
        f"{probability * 100:.2f}%"
    )


    st.progress(
        float(probability)
    )


    # ========================================================
    # PROBABILITY INTERPRETATION
    # ========================================================

    if probability < 0.30:

        st.info(
            "The model estimates a relatively low "
            "probability of the applicant belonging "
            "to the high-risk class."
        )


    elif probability < 0.70:

        st.warning(
            "The model estimates an intermediate "
            "probability of the applicant belonging "
            "to the high-risk class."
        )


    else:

        st.warning(
            "The model estimates a relatively high "
            "probability of the applicant belonging "
            "to the high-risk class."
        )


    # ========================================================
    # SAVE PREDICTION HISTORY
    # ========================================================

    st.session_state.prediction_history.append({

        "Model": selected_model,

        "Prediction":
            "High Risk"
            if prediction == 1
            else "Low Risk",

        "Risk Probability":
            f"{probability * 100:.2f}%"

    })



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💳 Credit Risk")


    st.write("### About")


    st.write(
        """
        This application predicts credit risk
        using machine learning classification models.
        """
    )


    st.divider()


    st.write("### Available Models")


    st.write(
        """
        - 🌲 Random Forest
        - 🚀 Gradient Boosting
        - ⚡ XGBoost
        """
    )


    st.divider()


    st.write("### Input Features")


    st.write(
        """
        - Person Age
        - Person Income
        - Home Ownership
        - Employment Length
        - Loan Intent
        - Loan Grade
        - Loan Amount
        - Interest Rate
        - Loan Percent Income
        - Previous Default
        - Credit History Length
        """
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    st.divider()

    st.subheader(
        "📜 Prediction History"
    )


    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )


    st.dataframe(
        history_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_data = history_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Prediction History",

        data=csv_data,

        file_name="credit_risk_predictions.csv",

        mime="text/csv",

        use_container_width=True
    )
    
    
