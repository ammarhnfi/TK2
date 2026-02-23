
import geopandas as gpd
import pandas as pd

# Load Data
DATA_PATH = r"d:\tk2\san_ref\data\assignment_2_covid\covid19_eng.gpkg"
gdf = gpd.read_file(DATA_PATH)

# Define Variables
y_var = "Human_health_and_social_work"
x_vars = [
    "Education",
    "Financial_and_insurance",
    "Information_and_communication",
    "Real_estate"
]

# Select columns and get first 5 rows
df_subset = gdf[[y_var] + x_vars].head(5)

# Rename columns for better table display if needed (optional, but good for LaTeX)
# For now, let's keep them as is or slightly shorten them for the table header
column_mapping = {
    "Human_health_and_social_work": "Health & Social",
    "Education": "Education",
    "Financial_and_insurance": "Financial",
    "Information_and_communication": "Info & Comm",
    "Real_estate": "Real Estate"
}
df_subset = df_subset.rename(columns=column_mapping)

# Convert to LaTeX
latex_table = df_subset.to_latex(index=False, caption="Sampel Data (5 Baris Pertama)", label="tab:dataset_sample")

print(latex_table)
