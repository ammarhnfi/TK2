import geopandas as gpd

DATA_PATH = r"d:\tk2\san_ref\data\assignment_2_covid\covid19_eng.gpkg"
gdf = gpd.read_file(DATA_PATH)

print("Columns:", gdf.columns.tolist())
print("First 2 rows:", gdf.head(2))
