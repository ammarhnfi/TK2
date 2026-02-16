import geopandas as gpd
import pandas as pd
import os

# Load Data
DATA_PATH = r"d:\tk2\san_ref\data\assignment_2_covid\covid19_eng.gpkg"
OUTPUT_CSV = r"d:\tk2\data_analysis2.csv"

print("Loading data from", DATA_PATH)
gdf = gpd.read_file(DATA_PATH)

# Define Variables used in analysis
y_var = "Human_health_and_social_work"
x_vars = [
    "Education",
    "Financial_and_insurance",
    "Information_and_communication",
    "Real_estate"
]

# Check if ctyua19nm exists, if not try to find similar name or use objectid
# Based on typical UK data and data_analysis.csv, ctyua19nm should be there.
# If not, I'll print available columns to debug.
if "ctyua19nm" not in gdf.columns:
    # Try finding typical name columns
    print("Warning: 'ctyua19nm' not found. Searching for similar columns...")
    name_cols = [c for c in gdf.columns if "nm" in c.lower() or "name" in c.lower()]
    print("Potential name columns:", name_cols)
    # Fallback to the first string column if needed, or error out
    # For now, let's assume it exists or use the first name col found
    if name_cols:
         location_col = name_cols[0]
         print(f"Using {location_col} as location column.")
    else:
         raise ValueError("Could not find location name column")
else:
    location_col = "ctyua19nm"

# Calculate Lat/Long from Centroids
print("Calculating centroids...")
# Reproject to WGS84 (EPSG:4326) for lat/long if not already
if gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

gdf['long'] = gdf.geometry.centroid.x
gdf['lat'] = gdf.geometry.centroid.y

# Select Columns
cols_to_keep = [location_col, 'lat', 'long', y_var] + x_vars

# Create Subset
df_subset = gdf[cols_to_keep].copy()

# Rename location col to ctyua19nm if different (to match request)
if location_col != "ctyua19nm":
    df_subset = df_subset.rename(columns={location_col: "ctyua19nm"})

# Drop NA (same as analysis)
print("Dropping NAs...")
df_subset = df_subset.dropna()

# Save to CSV
print(f"Saving to {OUTPUT_CSV}...")
df_subset.to_csv(OUTPUT_CSV, index=False)

# Display first 5 rows
print("\nFirst 5 rows of data_analysis2.csv:")
print(df_subset.head())
