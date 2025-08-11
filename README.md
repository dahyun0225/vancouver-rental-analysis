# CMPT 353 Final Project – Student Rental Price Analysis

## Overview
This project analyzes Metro Vancouver rental listings to determine where students are most likely to find affordable housing.  
It uses ETL, feature engineering, and machine learning (Random Forest regression) to explore price trends, location effects, and property features that affect rental affordability.

The analysis produces:
- Cleaned and preprocessed rental data
- Visualizations of price vs. distance, square footage, and location
- Feature importance rankings from the Random Forest model
- Predictions of rental prices for sample listings

---

## Required Libraries
Make sure you have Python 3.9+ installed, then install the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn geopy
```

---

## Data
The repository includes:
- `data/rentals_raw.csv` — Raw scraped rental listings  
- `data/rentals_clean.csv` — Cleaned dataset after ETL  
- `figs/` — Generated visualizations  
- `figs/rf_predictions_sample.csv` — Sample predictions from the model  

---

## How to Run

### 1. Data Cleaning & ETL
```bash
python clean_rentals.py
```
This script loads `rentals_raw.csv`, cleans and formats the data, and outputs `rentals_clean.csv`.

### 2. Analysis & Modeling
```bash
python analyze_students_rentals.py
```
This script loads `rentals_clean.csv`, trains a Random Forest regression model, generates visualizations, and outputs predictions.

**Outputs:**
- All generated plots will be saved in the `figs/` folder  
- Sample predictions will be saved as `figs/rf_predictions_sample.csv`
