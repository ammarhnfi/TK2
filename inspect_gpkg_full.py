import geopandas as gpd
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

DATA_PATH = r"d:\tk2\san_ref\data\assignment_2_covid\covid19_eng.gpkg"
gdf = gpd.read_file(DATA_PATH)

print("Columns:")
for col in gdf.columns:
    print(col)
