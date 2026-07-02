# 📄 Resume Section — Smart House Price Prediction System

---

## 1. Project Description (for Resume Projects Section)

**Smart House Price Prediction System** | Python · Django · Scikit-Learn · Bootstrap 5
*Full-Stack ML Web Application — Capstone Project*

Built an end-to-end house price prediction system using the California Housing Dataset
(20,640 records). Implemented a complete ML pipeline including data cleaning, IQR-based
outlier removal, feature engineering (11 features from 8 raw inputs), and trained three
regression models (Linear Regression, Random Forest, Gradient Boosting) with automatic
best-model selection via R² score. Deployed predictions through a responsive Django web
application with SQLite-backed prediction history, a real-time analytics dashboard with
Chart.js visualizations, confidence scoring, and an EDA visualization gallery.

**Best Model Achieved:** R² = 0.8685 · RMSE = 0.2996 · MAE = 0.2232

---

## 2. Key Skills Demonstrated

### Machine Learning & Data Science
- Supervised regression modeling (Linear, Ensemble, Boosting methods)
- Data preprocessing — missing value handling, outlier detection (IQR)
- Feature engineering — deriving new numerical features from raw inputs
- Model evaluation — R², RMSE, MAE, 5-fold cross-validation
- Exploratory Data Analysis (EDA) with correlation analysis
- Professional data visualization — Seaborn heatmaps, scatter plots, pair plots
- Model serialization and deployment with Joblib

### Web Development & Software Engineering
- Full-stack Django application (MVT architecture)
- Django ORM — SQLite database design and migrations
- Django forms with server-side validation and custom error handling
- Django template engine — template inheritance, custom template filters
- RESTful URL design and view-based routing
- Bootstrap 5 responsive UI with custom dark theme CSS
- Chart.js integration for real-time dashboard charts
- Static file management with WhiteNoise

### General Engineering
- Object-oriented Python design
- Git version control workflow
- Project documentation (README)
- Production-level project structure

---

## 3. Professional Resume Bullet Points

Use any 3 of the following in your resume:

**Option A** *(for ML-focused roles)*
> Engineered an end-to-end house price prediction system (R² = 0.87) by training and
> comparing Linear Regression, Random Forest, and Gradient Boosting models on 20,640
> records; implemented automatic best-model selection, 5-fold cross-validation, and
> Joblib-based model persistence.

**Option B** *(for full-stack / web roles)*
> Developed a full-stack ML web application using Django 4.2 and Bootstrap 5, featuring
> a real-time analytics dashboard (Chart.js), SQLite prediction history with pagination,
> confidence-scored results, and 5 professional EDA visualizations served from a
> responsive dark-themed UI.

**Option C** *(for data engineering / pipeline roles)*
> Built a production-grade ML pipeline — data ingestion → IQR outlier removal → feature
> engineering (11 features) → multi-model training → auto selection → Django deployment —
> achieving 86.85% variance explained (R²) on California housing price regression.

**Option D** *(concise single-line for skills/projects table)*
> House Price Prediction: Django · Scikit-Learn · Pandas · Bootstrap 5 · SQLite |
> R² 0.87 | 3-model auto-selection | real-time dashboard

---

## 4. Interview Talking Points

- **Why did you choose these 3 models?** — To compare a simple baseline (Linear Regression),
  a bagging ensemble (Random Forest), and a boosting ensemble (Gradient Boosting) to
  demonstrate understanding of the bias-variance tradeoff.

- **How did you handle overfitting?** — Used 5-fold cross-validation to estimate
  generalization performance beyond the test split; Random Forest used max_depth=12
  and min_samples_split to prevent overfit.

- **What is feature engineering doing here?** — rooms_per_person normalizes room count by
  occupancy density; bedrooms_ratio captures bedroom proportion; income_per_room combines
  economic and spatial signals — all three improve predictive signal.

- **Why Joblib over Pickle?** — Joblib is optimized for large numpy arrays (which scikit-learn
  models contain internally) and provides faster serialization/deserialization.

- **How does the confidence scoring work?** — Predictions within the dataset's typical range
  (0.5–4.0 × $100k) are flagged High Confidence; extrapolated values get Medium or Low.
