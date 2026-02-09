import geopandas as gpd
try:
    gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
    print("Columns:", gdf.columns)
    print("\nFirst 5 rows:")
    cols_to_check = ['ctyua19nm', 'lat', 'long']
    existing_cols = [c for c in cols_to_check if c in gdf.columns]
    print(gdf[existing_cols].head())
    
    if 'geometry' in gdf.columns:
        print("\nCRS:", gdf.crs)
except Exception as e:
    print(e)
