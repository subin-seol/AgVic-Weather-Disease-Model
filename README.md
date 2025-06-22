# Almond Hull Rot Risk Prediction

This project aims to assist almond growers and researchers in predicting the risk of **hull rot**, a significant fungal disease, using weather data and orchard-specific inputs. It provides an interactive **Streamlit web application** and a complete pipeline for data preprocessing, modeling, and visualization.

---

## Project Structure
```text
almond-hullrot-risk/
├── 01_preprocessing.ipynb          # Cleans and aggregates raw tree-level data
├── 02_correlation_analysis.ipynb   # Feature selection and relationship analysis
├── 03_model_training.ipynb         # Trains and evaluates Random Forest model
├── app.py                          # Streamlit web application
├── data/
│   ├── Almond disease survey data ALL.csv
│   └── HR_processed_data.csv       # Output of preprocessing
├── hr_risk_model.pkl               # Trained Random Forest model
├── encoder.pkl                     # OneHotEncoder for categorical features
```

---

## Overview

The tool allows users to:

- Input orchard name, almond variety, and current weather stats
- Get instant prediction of hull rot risk (Low / Medium / High)
- Visualize results via traffic light indicator and bar plots
- Download results for further analysis or decision-making

---

## Notebooks Explained

| Notebook | Purpose |
|----------|---------|
| `01_preprocessing.ipynb` | Cleans and aggregates raw tree-level almond disease data into a structured orchard-level dataset with HR incidence. |
| `02_correlation_analysis.ipynb` | Performs exploratory data analysis, correlation analysis, and identifies most impactful weather features. |
| `03_model_training.ipynb` | Trains a `RandomForestClassifier` using selected features and exports the model and encoder as `.pkl` files. |

---

## Running the App

1. **Install Requirements**  
Make sure you have Python 3.9+ and install dependencies:
pip install streamlit pandas scikit-learn joblib matplotlib seaborn

2. Ensure files are in the correct location
Place the following in the root directory: hr_risk_model.pkl, encoder.pkl

3. Run the App
streamlit run app.py
