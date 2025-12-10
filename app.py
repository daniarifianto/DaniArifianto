import streamlit as st
import numpy as np
import pandas as pd
import joblib, json

st.title("Prediksi Diabetes (Decision Tree + ADASYN)")

# Load pipeline
pipeline = joblib.load("dt_pipeline.sav")

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

inputs = []

st.subheader("Input Data Pasien")

for feat in feature_names:
    val = st.number_input(feat, value=0.0)
    inputs.append(val)

if st.button("Prediksi"):
    X = np.array([inputs])
    pred = pipeline.predict(X)[0]

    if pred == 1:
        st.error("HASIL: PASIEN BERISIKO DIABETES")
    else:
        st.success("HASIL: PASIEN TIDAK BERISIKO DIABETES")
