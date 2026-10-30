"""
Streamlit demo: predict a used car's asking price.

Run from the repo root (after running the notebook once to create models/):
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import add_features, MODEL_FEATURES

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "car_price_model.joblib")
    meta = json.loads((MODEL_DIR / "app_metadata.json").read_text())
    return model, meta


REF_YEAR = 2024   # ages in the training data are measured from 2024


def build_input(brand, model_name, age, km, transmission, fuel, owner, engine, is_4x4, seven_seats, description):
    """One-row DataFrame with the same engineered features used in training.

    The listing text is rebuilt in the same "Brand Model details, YEAR, Fuel" title
    format most training listings use, so the text-based features are extracted
    exactly as they were during training.
    """
    details = " ".join(filter(None, [
        f"{engine:.1f}" if engine else "", "4x4" if is_4x4 else "", "7 Seater" if seven_seats else "", description]))
    fuel_text = {"Hybrid/CNG": "CNG & Hybrids"}.get(fuel, fuel)
    title = f"{brand} {model_name} {details}, {REF_YEAR - age}, {fuel_text}"
    row = pd.DataFrame([{
        "Brand": brand, "model": model_name, "Age": age, "kmDriven": km,
        "Transmission": transmission, "FuelType": fuel, "Owner": owner,
        "AdditionInfo": title,
    }])
    return add_features(row)[MODEL_FEATURES]


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run the notebook first — it saves models/car_price_model.joblib.")
    st.stop()

st.title("🚗 Used Car Price Predictor")
st.caption(f"Model: {meta['model_name']} · typical error ±{meta['test_metrics']['Median % error']:.0f}% "
           f"(median, on held-out test data)")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(meta["brands_models"]))
    model_name = st.selectbox("Model", meta["brands_models"][brand])
    age = st.slider("Age (years)", 0, meta["max_age"], 5)
    km = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=50_000, step=5_000)
with col2:
    transmission = st.selectbox("Transmission", meta["transmissions"])
    fuel = st.selectbox("Fuel type", meta["fuel_types"])
    owner = st.selectbox("Owner", meta["owners"])
    engine = st.number_input("Engine size in litres (0 = unknown)", 0.0, 6.5, 0.0, 0.1)
    is_4x4 = st.checkbox("4x4 / AWD")
    seven_seats = st.checkbox("7+ seats")
description = st.text_input("Extra details (optional)", placeholder="e.g. well maintained, single owner")

if st.button("Estimate price", type="primary", use_container_width=True):
    X = build_input(brand, model_name, age, km, transmission, fuel, owner, engine, is_4x4, seven_seats, description)
    price = float(model.predict(X)[0])
    err = meta["test_metrics"]["Median % error"] / 100
    st.metric("Estimated asking price", f"₹{price:,.0f}")
    st.write(f"Likely range: **₹{price * (1 - err):,.0f} – ₹{price * (1 + err):,.0f}**")
