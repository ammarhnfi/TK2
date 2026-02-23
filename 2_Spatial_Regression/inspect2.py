import geopandas as gpd
df = gpd.read_file('d:/tk2/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
print("Covid related columns:")
print([c for c in df.columns if 'covid' in c.lower()])
print("Other interesting columns:")
print([c for c in df.columns if any(x in c.lower() for x in ['density', 'age', 'old', 'pop', 'illness', 'imd', 'crowd', 'health', 'income'])])
