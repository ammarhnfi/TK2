
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen
from spreg import OLS, ML_Lag, ML_Error
import os

# Set working directory and file paths
DATA_PATH = r"d:\tk2\san_ref\data\assignment_2_covid\covid19_eng.gpkg"
OUTPUT_DIR = r"d:\tk2\output_spatial"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load Data
print("Loading data...")
gdf = gpd.read_file(DATA_PATH)

# Define Variables
y_var = "Human_health_and_social_work"
x_vars = [
    "Education",
    "Financial_and_insurance",
    "Information_and_communication",
    "Real_estate"
]

# Subset and Drop NAs
gdf_subset = gdf[[y_var] + x_vars + ["geometry"]].dropna()
print(f"Data shape after subsetting: {gdf_subset.shape}")

# Standardization (Z-score)
# Important: spreg expects numpy arrays for X and y
y = gdf_subset[y_var].values.reshape(-1, 1)
X = gdf_subset[x_vars].values

# Standardize
y_mean, y_std = y.mean(), y.std()
X_mean, X_std = X.mean(axis=0), X.std(axis=0)

y_stdz = (y - y_mean) / y_std
X_stdz = (X - X_mean) / X_std

# Create Spatial Weights Matrix
print("Creating Spatial Weights Matrix...")
w = Queen.from_dataframe(gdf_subset)
w.transform = 'r'

# 1. OLS Estimation
print("Running OLS...")
ols = OLS(y_stdz, X_stdz, w=w, name_y=y_var, name_x=x_vars, name_ds='covid19_eng', spat_diag=True, moran=True)
print(ols.summary)
with open(os.path.join(OUTPUT_DIR, "ols_summary.txt"), "w") as f:
    f.write(ols.summary)

# 2. Spatial Lag Model (SAR) - ML
print("Running SAR (ML_Lag)...")
sar = ML_Lag(y_stdz, X_stdz, w=w, name_y=y_var, name_x=x_vars, name_ds='covid19_eng')
print(sar.summary)
with open(os.path.join(OUTPUT_DIR, "sar_summary.txt"), "w") as f:
    f.write(sar.summary)

# 3. Spatial Error Model (SEM) - ML
print("Running SEM (ML_Error)...")
sem = ML_Error(y_stdz, X_stdz, w=w, name_y=y_var, name_x=x_vars, name_ds='covid19_eng')
print(sem.summary)
with open(os.path.join(OUTPUT_DIR, "sem_summary.txt"), "w") as f:
    f.write(sem.summary)

# Compare Models (Simple print)
print("\n--- Model Comparison ---")
print(f"OLS AIC: {ols.aic:.4f}")
print(f"SAR AIC: {sar.aic:.4f}")
print(f"SEM AIC: {sem.aic:.4f}")

# Save comparison to file
with open(os.path.join(OUTPUT_DIR, "model_comparison.txt"), "w") as f:
    f.write(f"OLS AIC: {ols.aic:.4f}\n")
    f.write(f"SAR AIC: {sar.aic:.4f}\n")
    f.write(f"SEM AIC: {sem.aic:.4f}\n")
    best_model = "SAR" if sar.aic < sem.aic and sar.aic < ols.aic else "SEM" if sem.aic < sar.aic and sem.aic < ols.aic else "OLS"
    f.write(f"Best model based on AIC: {best_model}\n")


# Generate Residual maps
# Add residuals to dataframe
gdf_subset['ols_resid'] = ols.u
gdf_subset['sar_resid'] = sar.u
gdf_subset['sem_resid'] = sem.u

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

gdf_subset.plot(column='ols_resid', cmap='coolwarm', legend=True, ax=axes[0])
axes[0].set_title('OLS Residuals')
axes[0].axis('off')

gdf_subset.plot(column='sar_resid', cmap='coolwarm', legend=True, ax=axes[1])
axes[1].set_title('SAR Residuals')
axes[1].axis('off')

gdf_subset.plot(column='sem_resid', cmap='coolwarm', legend=True, ax=axes[2])
axes[2].set_title('SEM Residuals')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_comparison.png"))
plt.close()

print("Analysis complete. Results saved to d:\\tk2\\output_spatial")
