# AgriIntel — Rice Yield Failure Prediction

A machine learning system that predicts rice crop yield failure 
risk based on field-level conditions.

## What it does
Takes inputs — rainfall, fertilizer, pesticide, area, state, 
season — and outputs a failure probability with key risk drivers 
explained. Deployed as a live interactive Streamlit dashboard.

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 93% |
| ROC-AUC | 0.856 |
| Failure Recall | 64% |
| Failure Precision | 37% |

## Tech Stack
- Python · XGBoost · Scikit-learn · SMOTE
- Streamlit · Plotly · GeoPandas

## How to run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`

## Dataset
Indian agricultural rice data (1997–2020) used as a proxy for 
Nigerian conditions. Source: Kaggle.

## Note on GADM files
The map requires GADM India Level-1 shapefiles 
(gadm41_IND_1.*). Download from gadm.org and place in the 
same folder as app.py.
