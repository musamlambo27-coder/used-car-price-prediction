# Used Car Price Prediction

Capstone project: a machine-learning model that predicts the asking price of a used car from its listing (brand, model, age, kilometres, transmission, fuel type, owner, seller description).

**Author:** Musa Mlambo

## Results

| Model | Test RMSE (₹) | Test MAE (₹) | Test R² | Median % error |
|---|---|---|---|---|
| Baseline (predict mean) | 14,94,284 | 6,52,974 | −0.07 | 53.8% |
| Ridge regression | 7,05,927 | 2,24,808 | 0.76 | 14.0% |
| Random forest | 7,00,942 | 2,21,870 | 0.76 | 14.8% |
| Gradient boosting (default) | 6,91,770 | 2,19,305 | 0.77 | 14.5% |
| **Gradient boosting (tuned) — final** | **6,30,679** | **1,96,666** | **0.81** | **13.3%** |

**Key findings:**
- Asking prices fall about **12% per year** of age.
- Luxury brands list at about **4×** the median of other brands.
- Automatics have a median of ₹7.75 lakh vs ₹4.5 lakh for manuals.
- The seller's free text recovered ~98% of missing brands and years with >99% accuracy.

## Project structure

```
├── used_car_price_prediction.ipynb   # Full analysis: cleaning → EDA → features → models → tuning → error analysis
├── app.py                            # Streamlit demo app
├── src/
│   └── features.py                   # Parsing + feature engineering shared by notebook and app
├── data/
│   └── used_car_dataset.csv          # Raw course dataset
├── models/                           # Saved model + app metadata (created by the notebook)
├── reports/figures/                  # Charts saved by the notebook (used in slides)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
# 1. put the dataset at data/used_car_dataset.csv
# 2. run the notebook top to bottom (Kernel → Restart & Run All)
jupyter notebook used_car_price_prediction.ipynb
# 3. launch the demo app
streamlit run app.py
```

## Approach

1. **Cleaning** (24,575 raw rows → 13,591 unique listings):
   - parse price and km text into numbers;
   - turn placeholders ("Unknown", Year 9999) into missing values and merge duplicate fuel labels;
   - remove 9,592 exact duplicates and merge 1,227 near-duplicate copies;
   - **recover missing brand/model/year/fuel/transmission from the seller text**, verified against known values;
   - remove extreme price outliers (3×IQR on log-price) and non-car ads.
2. **EDA:** price distribution, depreciation by age, price by fuel/transmission/owner/brand, correlations.
3. **Feature engineering:**
   - added `km_per_year`, `engine_litres` (parsed from the title), `is_luxury`, `is_4x4`, `is_7_seater`, `is_dealer_listing` and `mentions_good_condition`;
   - dropped flags that are too rare to learn from, and `Year` (a duplicate of `Age`).
4. **Modelling:** mean baseline, Ridge, Random Forest, Histogram Gradient Boosting — all in one scikit-learn pipeline, trained on log(price), compared with 5-fold CV and a 20% hold-out test set.
5. **Tuning:** `RandomizedSearchCV` on gradient boosting.
6. **Error analysis:** predicted vs actual, error by price band and brand, worst predictions, train-vs-validation gap, learning curve, permutation importance.
