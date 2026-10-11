
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Classification")

st.title("🩺 Diabetes Classification")
st.write("Enter the patient's details to get a model prediction.")

try:
    model = joblib.load("best_model.pkl")
    feature_names = joblib.load("features.pkl")
except Exception as e:
    st.error("Model files could not be loaded.")
    st.exception(e)
    st.stop()

values = {}

for feature in feature_names:
    values[feature] = st.number_input(
        feature,
        min_value=0.0,
        value=0.0,
        step=1.0
    )

if st.button("Predict"):
    try:
        input_data = pd.DataFrame(
            [values],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.warning(
                "The model predicts a positive diabetes classification."
            )
        else:
            st.success(
                "The model predicts a negative diabetes classification."
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.caption(
    "Educational demonstration only. This prediction is not a medical diagnosis."
)

