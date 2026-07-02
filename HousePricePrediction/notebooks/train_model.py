"""
train_model.py
Complete ML Pipeline: Data Loading → EDA → Feature Engineering →
Model Training → Evaluation → Visualization → Model Saving
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MEDIA_DIR  = os.path.join(BASE_DIR, 'media')
VIZ_DIR    = os.path.join(MEDIA_DIR, 'visualizations')

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(VIZ_DIR,    exist_ok=True)

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor':   '#1e293b',
    'axes.edgecolor':   '#334155',
    'axes.labelcolor':  '#e2e8f0',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#e2e8f0',
    'grid.color':       '#334155',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT      = '#38bdf8'
ACCENT2     = '#818cf8'
ACCENT3     = '#34d399'


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SMART HOUSE PRICE PREDICTION — ML PIPELINE")
print("═"*60)

sys.path.insert(0, os.path.join(BASE_DIR, 'notebooks'))
from preprocessing import (load_dataset, handle_missing_values,
                            remove_outliers, engineer_features,
                            get_feature_target_split, scale_features,
                            FEATURE_NAMES)

print("\n[1/8] Loading California Housing Dataset …")
df = load_dataset()
print(f"      Shape: {df.shape}")
print(df.head())


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Data Cleaning …")
df = handle_missing_values(df)
df = remove_outliers(df)
print(f"      Clean shape: {df.shape}")
print(df.describe().round(3))


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Feature Engineering …")
df = engineer_features(df)
print("      New features: rooms_per_person, bedrooms_ratio, income_per_room")

X, y = get_feature_target_split(df)
print(f"      Features: {list(X.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT  +  SCALING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] Train-Test Split & Scaling …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPLORATORY DATA ANALYSIS  +  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] EDA & Visualizations …")

# ── 5-a  Correlation Heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[FEATURE_NAMES + ['median_house_value']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=.5,
            cbar_kws={'shrink': .8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold',
             color=ACCENT, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print("      ✓ correlation_heatmap.png")

# ── 5-b  Price Distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('House Price Distribution', fontsize=16,
             fontweight='bold', color=ACCENT)

axes[0].hist(y, bins=50, color=ACCENT, edgecolor='#0f172a', alpha=0.85)
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Raw Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].hist(np.log1p(y), bins=50, color=ACCENT2, edgecolor='#0f172a',
             alpha=0.85)
axes[1].set_xlabel('log(Median House Value + 1)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Log-Transformed Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'price_distribution.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print("      ✓ price_distribution.png")

# ── 5-c  Scatter Plots ────────────────────────────────────────────────────────
top_features = ['median_income', 'avg_rooms', 'house_age', 'avg_occupancy']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature vs. House Price Scatter Plots', fontsize=16,
             fontweight='bold', color=ACCENT)
axes = axes.flatten()
for i, feat in enumerate(top_features):
    axes[i].scatter(df[feat], df['median_house_value'],
                    alpha=0.3, s=5, color=ACCENT if i % 2 == 0 else ACCENT2)
    axes[i].set_xlabel(feat.replace('_', ' ').title())
    axes[i].set_ylabel('Median House Value')
    axes[i].set_title(f'{feat.replace("_"," ").title()} vs Price')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'scatter_plots.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print("      ✓ scatter_plots.png")

# ── 5-d  Pair Plot (subset) ───────────────────────────────────────────────────
pair_cols = ['median_income', 'avg_rooms', 'house_age', 'median_house_value']
pair_df   = df[pair_cols].sample(n=1500, random_state=42)

pair_fig = plt.figure(figsize=(12, 12))
pair_fig.patch.set_facecolor('#0f172a')
pg = sns.pairplot(pair_df, plot_kws={'alpha': 0.4, 's': 10, 'color': ACCENT},
                  diag_kws={'color': ACCENT2, 'edgecolor': '#0f172a'})

for ax in pg.axes.flatten():
    if ax:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#e2e8f0')
        ax.yaxis.label.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

pg.fig.suptitle('Pair Plot — Key Features', y=1.01, fontsize=16,
                fontweight='bold', color=ACCENT)
plt.savefig(os.path.join(VIZ_DIR, 'pair_plot.png'), dpi=100,
            bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("      ✓ pair_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Training Models …")

models = {
    'Linear Regression':        LinearRegression(),
    'Random Forest':            RandomForestRegressor(
                                    n_estimators=200,
                                    max_depth=15,
                                    min_samples_split=4,
                                    random_state=42,
                                    n_jobs=-1),
    'Gradient Boosting':        GradientBoostingRegressor(
                                    n_estimators=200,
                                    learning_rate=0.08,
                                    max_depth=5,
                                    random_state=42),
}

results = {}
trained = {}

for name, model in models.items():
    print(f"\n  ▸ {name}")
    model.fit(X_train_sc, y_train)
    preds        = model.predict(X_test_sc)
    r2           = r2_score(y_test, preds)
    rmse         = np.sqrt(mean_squared_error(y_test, preds))
    mae          = mean_absolute_error(y_test, preds)
    cv_scores    = cross_val_score(model, X_train_sc, y_train,
                                   cv=5, scoring='r2', n_jobs=-1)
    results[name] = {
        'R2 Score':  round(r2,   4),
        'RMSE':      round(rmse, 4),
        'MAE':       round(mae,  4),
        'CV Mean R2': round(cv_scores.mean(), 4),
        'CV Std':    round(cv_scores.std(),  4),
    }
    trained[name] = model
    print(f"    R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON  +  BEST MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Model Comparison …")
results_df = pd.DataFrame(results).T
print("\n" + results_df.to_string())

best_name  = results_df['R2 Score'].idxmax()
best_model = trained[best_name]
print(f"\n  ★  Best Model: {best_name}  "
      f"(R²={results[best_name]['R2 Score']})")


# ── Feature Importance ────────────────────────────────────────────────────────
feat_names_pretty = [f.replace('_', ' ').title() for f in FEATURE_NAMES]

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    # Use permutation importance surrogate for linear model
    importances = np.abs(best_model.coef_)

sorted_idx  = np.argsort(importances)[::-1]
sorted_imp  = importances[sorted_idx]
sorted_feat = [feat_names_pretty[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(12, 7))
colors  = [ACCENT if i == 0 else ACCENT2 if i == 1 else ACCENT3
           if i == 2 else '#64748b' for i in range(len(sorted_feat))]
bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1],
               edgecolor='#0f172a', height=0.6)

for bar, val in zip(bars, sorted_imp[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', color='#e2e8f0',
            fontsize=9)

ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title(f'Feature Importance — {best_name}', fontsize=15,
             fontweight='bold', color=ACCENT, pad=15)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print("      ✓ feature_importance.png")

# ── Model Comparison Bar Chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold', color=ACCENT)
palette = [ACCENT, ACCENT2, ACCENT3]
metrics_plot = ['R2 Score', 'RMSE', 'MAE']
names = list(results.keys())

for i, metric in enumerate(metrics_plot):
    vals = [results[n][metric] for n in names]
    axes[i].bar(names, vals, color=palette, edgecolor='#0f172a', width=0.5)
    axes[i].set_title(metric, color=ACCENT2, fontsize=13)
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=12)
    axes[i].grid(True, axis='y', alpha=0.3)
    for j, v in enumerate(vals):
        axes[i].text(j, v + max(vals)*0.01, f'{v:.3f}', ha='center',
                     color='#e2e8f0', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_comparison.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print("      ✓ model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE MODEL + ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Saving Model Artifacts …")

joblib.dump(best_model,          os.path.join(MODEL_DIR, 'best_model.pkl'))
joblib.dump(scaler,              os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(results,             os.path.join(MODEL_DIR, 'model_results.pkl'))
joblib.dump(FEATURE_NAMES,       os.path.join(MODEL_DIR, 'feature_names.pkl'))
joblib.dump({'name': best_name,
             'metrics': results[best_name]},
            os.path.join(MODEL_DIR, 'best_model_info.pkl'))

print(f"      ✓ best_model.pkl      → {best_name}")
print(f"      ✓ scaler.pkl")
print(f"      ✓ model_results.pkl")
print(f"      ✓ feature_names.pkl")
print(f"      ✓ best_model_info.pkl")

print("\n" + "═"*60)
print("  PIPELINE COMPLETE")
print("═"*60 + "\n")
