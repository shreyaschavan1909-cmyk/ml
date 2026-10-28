from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = Path(__file__).parent / "Automobile_data.csv"
NUM = ["symboling", "normalized-losses", "width", "height",
       "engine-size", "horsepower", "city-mpg", "highway-mpg"]
CAT = ["make", "fuel-type", "body-style", "drive-wheels",
       "engine-location", "engine-type"]


@st.cache_resource
def train():
    d = pd.read_csv(CSV).replace("?", np.nan)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(d[c].median())
    for c in NUM + ["price"]:                       # remove outliers
        q1, q3 = d[c].quantile(0.25), d[c].quantile(0.75)
        iqr = q3 - q1
        d = d[(d[c] >= q1 - 1.5 * iqr) & (d[c] <= q3 + 1.5 * iqr)]
    X = d[NUM + CAT]
    model = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])),
        ("lasso", Lasso(alpha=50, max_iter=20000)),
    ]).fit(X, d["price"])
    return model, d


model, data = train()

st.title("🚗 Car Price Predictor")

inputs = {}
for c in CAT:
    inputs[c] = st.selectbox(c.replace("-", " ").title(), sorted(data[c].unique()))
for c in NUM:
    inputs[c] = st.number_input(c.replace("-", " ").title(),
                                value=float(data[c].median()))

if st.button("Predict price"):
    price = model.predict(pd.DataFrame([inputs]))[0]
    st.success(f"Estimated price: {price:,.0f}")