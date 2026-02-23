
import geopandas as gpd
import pandas as pd

try:
    gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
    with open('cols.txt', 'w') as f:
        for col in gdf.columns:
            f.write(col + '\n')
    print("Columns written to cols.txt")
except Exception as e:
    with open('cols.txt', 'w') as f:
        f.write(str(e))
