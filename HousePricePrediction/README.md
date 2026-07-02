# 🏠 Smart House Price Prediction System

> **A production-grade Machine Learning + Django Web Application Capstone Project**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=flat&logo=django&logoColor=white)](https://djangoproject.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=flat&logo=bootstrap&logoColor=white)](https://getbootstrap.com)

---

## 📋 Project Overview

The **Smart House Price Prediction System** is a full-stack ML-powered web application that predicts California house prices in real time. It features a complete data science pipeline — from raw data ingestion through EDA, feature engineering, multi-model training with automatic best-model selection — served through a responsive Django web interface with persistent prediction history and an analytics dashboard.

---

## ✨ Features

### Machine Learning Pipeline
- ✅ Loads **California Housing Dataset** (20,640 records)
- ✅ Data cleaning — missing value handling, IQR-based outlier removal
- ✅ **Feature Engineering** — 3 new derived features (rooms_per_person, bedrooms_ratio, income_per_room)
- ✅ **5 professional EDA visualizations** — Correlation Heatmap, Price Distribution, Scatter Plots, Pair Plot, Feature Importance
- ✅ Trains and evaluates **3 ML models** — Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- ✅ **Auto model selection** based on R² score
- ✅ **5-fold cross-validation** for each model
- ✅ Evaluation metrics: R², RMSE, MAE
- ✅ Model + scaler saved with **Joblib**

### Django Web Application
- ✅ **Home Page** — Hero section, model comparison cards, EDA visualizations
- ✅ **Predict Page** — 8-input form with validation, tooltips & sample data
- ✅ **Result Page** — Price display, confidence badge, input summary, price gauge
- ✅ **Dashboard** — KPI cards, Chart.js charts, model comparison table
- ✅ **Prediction History** — Paginated SQLite records
- ✅ **About Page** — Pipeline walkthrough, tech stack, model evaluation
- ✅ **Responsive Bootstrap 5** dark theme UI
- ✅ **Confidence Scoring** — High / Medium / Low per prediction
- ✅ **Admin Panel** — Full CRUD on prediction records

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Django 4.2 |
| ML Library | Scikit-Learn 1.3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| UI Framework | Bootstrap 5.3 |
| Charts | Chart.js 4.4 |
| Database | SQLite (via Django ORM) |
| Model Persistence | Joblib |
| Static Files | WhiteNoise |

---

## 📊 Model Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | **0.8685** ⭐ | 0.2996 | 0.2232 |
| Random Forest | 0.8639 | 0.3048 | 0.2189 |
| Gradient Boosting | 0.8679 | 0.3003 | 0.2201 |

> Best model is **automatically selected** at runtime based on R² score.

---

## 📁 Project Structure

```
HousePricePrediction/
│
├── notebooks/
│   ├── train_model.py          # Complete ML pipeline
│   └── preprocessing.py        # Data preprocessing utilities
│
├── dataset/
│   └── california_housing.csv  # Raw dataset (20,640 records)
│
├── model/
│   ├── best_model.pkl          # Serialized best model
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── model_results.pkl       # All model evaluation metrics
│   ├── feature_names.pkl       # Feature name list
│   └── best_model_info.pkl     # Best model name + metrics
│
├── predictor/                  # Django app
│   ├── migrations/
│   ├── templatetags/
│   │   └── custom_filters.py   # Django split/index filters
│   ├── models.py               # PredictionRecord model
│   ├── views.py                # All page views
│   ├── forms.py                # PredictionForm with validation
│   ├── urls.py                 # App URL patterns
│   ├── admin.py                # Admin customization
│   └── apps.py
│
├── templates/
│   └── predictor/
│       ├── base.html           # Base layout with navbar/footer
│       ├── home.html           # Landing page
│       ├── predict.html        # Prediction form
│       ├── result.html         # Prediction result
│       ├── dashboard.html      # Analytics dashboard
│       ├── history.html        # Prediction history
│       └── about.html          # Project info
│
├── static/
│   └── css/
│       └── main.css            # Full dark theme CSS
│
├── media/
│   └── visualizations/         # Generated plots (PNG)
│       ├── correlation_heatmap.png
│       ├── price_distribution.png
│       ├── feature_importance.png
│       ├── scatter_plots.png
│       └── pair_plot.png
│
├── house_prediction/           # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── db.sqlite3                  # SQLite database
├── manage.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/HousePricePrediction.git
cd HousePricePrediction
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
```bash
python notebooks/train_model.py
```
This will:
- Load and clean the dataset
- Engineer features
- Train 3 ML models
- Save the best model + scaler to `model/`
- Generate 5 visualization PNGs to `media/visualizations/`

### 5. Run Django Migrations
```bash
python manage.py migrate
```

### 6. (Optional) Create Admin User
```bash
python manage.py createsuperuser
```

### 7. Run the Development Server
```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 📸 Screenshots

| Page | Description |
|---|---|
| **Home** | Hero section, model comparison, EDA visualizations |
| **Predict** | 8-field form with validation & location shortcuts |
| **Result** | Price in large display, confidence badge, input summary |
| **Dashboard** | KPI cards, Chart.js bar + donut charts |
| **History** | Paginated table of all predictions |
| **About** | Pipeline steps, tech stack, model table |

---

## 🔭 Input Features

| Feature | Description | Range |
|---|---|---|
| Median Income | Block median income ($10k units) | 0.5 – 15.0 |
| House Age | Median house age in block | 1 – 52 years |
| Avg Rooms | Average rooms per dwelling | 1.0 – 15.0 |
| Avg Bedrooms | Average bedrooms per dwelling | 1.0 – 5.0 |
| Population | Total block population | 3 – 35,682 |
| Avg Occupancy | Average persons per house | 0.5 – 20.0 |
| Latitude | Block latitude | 32.5° – 42.0° |
| Longitude | Block longitude | -124.5° – -114.5° |

**Engineered Features** (auto-computed internally):
- `rooms_per_person` = avg_rooms / avg_occupancy
- `bedrooms_ratio` = avg_bedrooms / avg_rooms
- `income_per_room` = median_income / avg_rooms

---

## 🔮 Future Improvements

- [ ] Add XGBoost and LightGBM models
- [ ] Implement SHAP explainability charts
- [ ] Add user authentication for personalized history
- [ ] Deploy to Heroku / Railway / AWS EC2
- [ ] Add API endpoint (`/api/predict/`) with DRF
- [ ] Export predictions to CSV
- [ ] Add geographic map visualization (Folium/Leaflet)
- [ ] Model retraining trigger from admin panel

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.

---

## 👨‍💻 Author

Built as a **Capstone Project** demonstrating full-stack ML engineering skills:
Python · Django · Scikit-Learn · Pandas · NumPy · Matplotlib · Seaborn · Bootstrap 5
