
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

try:
    print("Loading data...")
    gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
    
    # Calculate variables
    gdf['covid19_r'] = (gdf['X2020.04.14'] / gdf['Residents']) * 100000
    gdf['ethnic'] = (gdf['Mixed'] + gdf['Indian'] + gdf['Pakistani'] + gdf['Bangladeshi'] + 
                   gdf['Chinese'] + gdf['Other_Asian'] + gdf['Black'] + gdf['Other_ethnicity']) / gdf['Residents']
    gdf['lt_illness'] = gdf['Long_term_ill'] / gdf['Residents']
    
    gdf = gdf.fillna(0)
    
    # OLS
    X = gdf[['ethnic', 'lt_illness']]
    y = gdf['covid19_r']
    X_ols = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols).fit()
    
    # Map Residuals
    print("Mapping OLS Residuals...")
    gdf['ols_resid'] = ols_model.resid
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(column='ols_resid', ax=ax, legend=True,
             legend_kwds={'title': "Residuals", 'loc': 'lower right'},
             cmap='RdBu', scheme='quantiles')
    plt.title('Peta Residual OLS')
    plt.savefig('d:/tk1/figs/map_ols_residuals.png')
    plt.close()
    print("Map generated successfully: d:/tk1/figs/map_ols_residuals.png")

except Exception as e:
    print(f"Error: {e}")
