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
May, 2025

## 🇰🇷 한국어 버전 README

## 개요 
이 프로젝트는 **밴쿠버 지역의 학생 임대 시장(Student Rental Market)** 을 분석하여,  
어떤 지역에서 학생들이 가장 합리적인 가격의 주거를 찾을 수 있는지 탐색하는 데이터 분석 프로젝트입니다.

크롤링 → 전처리 → 지리적 거리 계산 → 시각화 → 랜덤 포레스트 모델링까지  
**End-to-End 데이터 파이프라인 전체를 직접 구축**하였으며,  
지역별 접근성, 면적, 편의시설 여부 등이 **임대료에 어떤 영향을 미치는지 정량적으로 분석**했습니다.

---

## 주요 기능 (Key Features)

### 1. 데이터 수집 & 전처리
- Craigslist 원본 데이터를 직접 수집  
- 주소 정규화, 위경도 변환  
- 이상치 처리 및 결측치 정리  
- 학생 전용 매물(student_flag) 파생 변수 생성

### 2. 지리 정보 기반 Feature Engineering
- 주요 지역(SFU, UBC, Downtown)까지의 거리 계산  
- 최근접 주요 지역(nearest_area) 자동 분류  
- 대중교통 접근성(near_transit) 변수 생성  
- 면적, 가구 포함 여부(furnished), 주차 여부 등 다양한 피처 정리

### 3. 시각화 (Visualization)
- 임대료 분포 히스토그램  
- 지리적 위치 시각화 (scatter_geo)  
- 임대료 vs 거리 관계 그래프  
- 임대료 vs 면적 관계 그래프

### 4. 머신러닝 모델링 (Random Forest Regressor)
- 임대 가격 예측 모델 구축  
- 가장 영향력 있는 변수 도출 (lat, distance_to_nearest_km 등)  
- 예측값 샘플 출력

---

## 데이터 파이프라인 (Data Pipeline)

1. **craigslist_scraper_enriched.py**  
   - Craigslist에서 크롤링  
   - 주소/위치 정보 보정  
   - 지리적 변수 생성 후 CSV 저장

2. **clean_rentals.py**  
   - 이상치/결측치 처리  
   - 카테고리/숫자형 정리  
   - 분석용 클린 데이터 생성

3. **analyze_students_rentals.py**  
   - EDA(탐색적 분석)  
   - 지리적 시각화  
   - Random Forest 회귀 모델  
   - Feature importance 산출  
   - 예측 결과 생성

---

## 결과 요약

- **위도(lat)** 와 특정 지역까지의 거리(SFU/UBC/Downtown)가  
  임대료를 설명하는 가장 중요한 요인으로 나타남.
- SFU 인근 매물은 **가격 변동 폭이 큼**  
- Downtown은 **고가 매물이 집중**, UBC 인근은 **거리가 분석에 크게 기여**  
- 지리 기반 feature가 모델 성능 향상에 핵심적으로 작용함.

---

## 프로젝트 구조

```
vancouver-rental-analysis/
│
├── data/                       # 정제된 원본/클린 데이터
├── figs/                       # 시각화 이미지 출력
├── src/                        # 파이프라인 코드
│   ├── craigslist_scraper_enriched.py
│   ├── clean_rentals.py
│   └── analyze_students_rentals.py
│
├── README.md                   # 프로젝트 설명
└── Vancouver rental project.pdf  # 최종 분석 보고서
```

---

## 한 줄 요약

“밴쿠버 학생 주거 문제를 데이터로 재구성하고,  
지리 정보 + 머신러닝 모델링으로 임대료 패턴을 정량적으로 분석한 프로젝트입니다.”
