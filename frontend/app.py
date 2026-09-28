import streamlit as st
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "AI-powered telecom customer churn prediction "
    "using an Artificial Neural Network."
)


st.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.subheader("Customer Information")


tenure = st.number_input(
    "Tenure (months)",
    min_value=0.0,
    max_value=100.0,
    value=12.0,
    step=1.0
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=70.0,
    step=1.0
)


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=100000.0,
    value=840.0,
    step=10.0
)


senior_citizen = st.selectbox(
    "Senior Citizen",
    options=[0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # FastAPI URL
    api_url = "http://127.0.0.1:8000/predict"


    # Data sent to FastAPI
    payload = {

        "tenure": tenure,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges,

        "SeniorCitizen": senior_citizen
    }


    try:

        response = requests.post(
            api_url,
            json=payload
        )


        if response.status_code == 200:

            result = response.json()


            probability = result[
                "churn_probability"
            ]

            prediction = result[
                "prediction"
            ]

            message = result[
                "result"
            ]


            st.divider()

            st.subheader(
                "Prediction Result"
            )


            # Probability
            st.metric(
                "Churn Probability",
                f"{probability}%"
            )


            # Result
            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is likely to STAY"
                )


            # Progress bar
            st.progress(
                min(
                    int(probability),
                    100
                )
            )


        else:

            st.error(
                f"API Error: {response.text}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI. "
            "Make sure the backend is running."
        )