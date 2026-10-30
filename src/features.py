"""
Shared data-parsing and feature-engineering helpers.

Used by BOTH the notebook (training) and app.py (Streamlit demo), so the
model always sees features built exactly the same way.
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "AskPrice"

LUXURY_BRANDS = {
    "audi", "bmw", "mercedes-benz", "mercedes", "jaguar", "land rover", "lexus",
    "porsche", "volvo", "mini", "jeep", "maserati", "bentley", "rolls-royce",
    "lamborghini", "ferrari",
}

# Candidate keyword flags extracted from the free-text AdditionInfo column.
# All are created and tested in the notebook; only the useful ones are kept below.
TEXT_FLAGS = {
    "is_4x4": r"4x4|4wd|\bawd\b|4 wd|quattro|xdrive|4matic|4motion",
    "is_7_seater": r"\b[789]\s*-?\s*seat|\b[789]\s*str\b",
    "is_dealer_listing": r",\s*(?:19|20)\d\d\s*,\s*(?:petrol|diesel|cng & hybrids|electric|lpg)\s*$",
    "mentions_good_condition": r"mint|excellent|good condition|well[- ]maintained|scratch[- ]?less",
    "mentions_service_history": r"servic",
    "mentions_insurance": r"insurance|insured",
    "mentions_accident": r"accident",
}
# Engine size in litres, e.g. "Innova 2.5 G" -> 2.5
ENGINE_PATTERN = r"\b([0-6]\.\d)(?![\d.])(?!\s*(?:lakh|lac|cr|k\b|km))"

# Final feature set fed to the model (selection justified in notebook Section 5)
NUMERIC_FEATURES = ["Age", "kmDriven", "km_per_year", "engine_litres"]
CATEGORICAL_FEATURES = ["Brand", "model", "Transmission", "FuelType", "Owner"]
BINARY_FEATURES = ["is_luxury", "is_4x4", "is_7_seater", "is_dealer_listing", "mentions_good_condition"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


# ---------------------------------------------------------------------------
# Parsing helpers (raw strings -> clean types)
# ---------------------------------------------------------------------------
def to_number(series: pd.Series) -> pd.Series:
    """Convert strings like '₹ 1,95,000', '98,000 km' or '4.5 Lakh' to floats."""
    s = series.astype(str).str.lower().str.strip()
    multiplier = np.where(s.str.contains("crore"), 1e7,
                          np.where(s.str.contains("lakh|lac"), 1e5, 1.0))
    digits = s.str.replace(r"[^\d.]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(digits, errors="coerce") * multiplier


def parse_posted_date(series: pd.Series) -> pd.Series:
    """Parse dates such as 'Nov-24'; fall back to pandas' general parser."""
    parsed = pd.to_datetime(series, format="%b-%y", errors="coerce")
    fallback = pd.to_datetime(series[parsed.isna()], errors="coerce")
    return parsed.fillna(fallback)


def standardise_text(series: pd.Series) -> pd.Series:
    """Trim, collapse spaces, and map case variants to their most common spelling."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    canonical = (s.dropna().groupby(s.dropna().str.lower())
                 .agg(lambda x: x.value_counts().index[0]))
    result = s.str.lower().map(canonical).astype("object")
    return result.where(result.notna(), np.nan)   # plain NaN, not pd.NA (scikit-learn needs NaN)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added."""
    out = df.copy()

    # scikit-learn imputers need plain NaN in text columns, not pandas' pd.NA
    for col in CATEGORICAL_FEATURES:
        if col in out:
            out[col] = out[col].astype("object").where(out[col].notna(), np.nan)

    # Usage intensity: how hard was the car driven per year of life?
    out["km_per_year"] = out["kmDriven"] / out["Age"].clip(lower=1)

    # Premium-brand indicator
    out["is_luxury"] = out["Brand"].astype(str).str.lower().isin(LUXURY_BRANDS).astype(int)

    # Keyword flags from the seller's free-text description
    text = out.get("AdditionInfo", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    for name, pattern in TEXT_FLAGS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(int)

    # Engine size (litres) when the title states it; missing otherwise (imputed in the pipeline)
    engine = pd.to_numeric(text.str.extract(ENGINE_PATTERN)[0], errors="coerce")
    out["engine_litres"] = engine.where(engine.between(0.6, 6.5))

    return out
