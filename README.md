# Vancouver Student Rental Price Analysis

This project analyzes Metro Vancouver rental listings to understand **where students are most likely to find affordable housing** and how location, transit access, and unit features affect monthly rent.

The project was designed and implemented as a full **end-to-end data science pipeline**, from web scraping to feature engineering, modeling, and visualization.

---

## 1. Project Overview

### **Goal**
- Build a clean rental dataset from raw Craigslist listings
- Quantify how **distance to SFU / UBC / Downtown**, transit access, and unit features relate to rent
- Provide tools for understanding **affordable student‑friendly areas** in the Vancouver region

### **Main Outputs**
- Fully cleaned, enriched rental dataset (CSV)
- Visualizations of:
  - Rent distribution (≤ $2,500)
  - Rent vs. distance to major areas (SFU, UBC, Downtown)
  - Rent vs. unit size
  - Geographic scatter map of listings
- Random Forest model identifying key drivers of rental prices

---

## 2. Data & ETL Pipeline

### **Source**
Raw rental listings scraped from **Craigslist Vancouver**.

### **Steps**
1. **Scraping**  
   `src/craigslist_scraper_enriched.py`  
   - HTML extraction of title, price, description, coordinates, amenities

2. **Cleaning & Enrichment**  
   `src/clean_rentals.py`  
   - Remove duplicates & invalid rows
   - Normalize price, square footage
   - Clean coordinate fields
   - Compute distances to:
     - SFU Burnaby
     - UBC Vancouver
     - Downtown Vancouver
   - Create feature flags:
     - `near_transit`
     - `student_flag`
     - `furnished`
     - `pets_allowed`
     - `parking_available`

3. **Analysis‑Ready Dataset**  
   - `data/rentals_clean.csv` (main cleaned dataset)
   - `data/rental_summary_combined.csv` (summary stats)

---

## 3. Analysis & Modeling

Main analysis script:  
`src/analyze_students_rentals.py`

### **Techniques Used**
- Exploratory data analysis (pandas, matplotlib)
- Feature engineering:
  - Distance to nearest major area
  - Unit size & amenities
  - Transit access
- Machine learning:
  - **Random Forest Regression** for predicting rent

### **Figures Generated**
Saved in `figs/`:

- `hist_price.png` — Rent distribution (≤ $2,500)
- `scatter_geo.png` — Map of listings by nearest area
- `scatter_price_vs_distance.png` — Rent vs. distance
- `scatter_price_vs_sqft.png` — Rent vs. square footage
- `random_forest_importances.png` — Feature importance ranking

---

## 4. Key Insights

- Rent around SFU, UBC, and Downtown remains high; **distance alone does not guarantee affordability**.
- For mid‑range units, **square footage and exact neighborhood** are often stronger predictors than distance.
- Transit access remains an influential factor — listings near major bus/skytrain routes show consistently higher prices.
- Random Forest results confirm that **location + accessibility + unit size** jointly shape rental affordability.

Additional narrative and interpretations are included in:  
`Vancouver rental project.pdf`

---

## 5. Repository Structure

```
vancouver-rental-analysis/
├── data/
│   ├── rentals_raw.csv
│   ├── rentals_clean.csv
│   └── rental_summary_combined.csv
├── figs/
│   ├── hist_price.png
│   ├── random_forest_importances.png
│   ├── scatter_geo.png
│   ├── scatter_price_vs_distance.png
│   └── scatter_price_vs_sqft.png
├── src/
│   ├── craigslist_scraper_enriched.py
│   ├── clean_rentals.py
│   └── analyze_students_rentals.py
├── Vancouver rental project.pdf
└── README.md
```

---

## 6. How to Run

```bash
# 1. (Optional) Re-scrape data
python src/craigslist_scraper_enriched.py

# 2. Clean and enrich scraped data
python src/clean_rentals.py

# 3. Run analysis & generate all plots
python src/analyze_students_rentals.py
```

### **Dependencies**
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- geopy (or similar geocoding library)

---

## 7. Author

**Dahyeon Choi**  
Data Science  
Vancouver, 2025
