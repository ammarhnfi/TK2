
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import numpy as np

try:
    print("Loading data...")
    gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
    
    # Calculate variables
    gdf['covid19_r'] = (gdf['X2020.04.14'] / gdf['Residents']) * 100000
    gdf['ethnic'] = (gdf['Mixed'] + gdf['Indian'] + gdf['Pakistani'] + gdf['Bangladeshi'] + 
                   gdf['Chinese'] + gdf['Other_Asian'] + gdf['Black'] + gdf['Other_ethnicity']) / gdf['Residents']
    gdf['lt_illness'] = gdf['Long_term_ill'] / gdf['Residents']
    
    gdf = gdf.fillna(0)
    gdf_clean = gdf[['covid19_r', 'ethnic', 'lt_illness', 'geometry']].copy()
    
    X = gdf_clean[['ethnic', 'lt_illness']].values
    y = gdf_clean['covid19_r'].values.reshape(-1, 1)
    coords = list(zip(gdf_clean.geometry.centroid.x, gdf_clean.geometry.centroid.y))
    
    print("Selecting bandwidth...")
    gwr_selector = Sel_BW(coords, y, X, kernel='bisquare', fixed=False)
    bw = gwr_selector.search(criterion='AICc')
    print(f"Optimal bandwidth: {bw}")
    
    print("Fitting GWR...")
    gwr_model = GWR(coords, y, X, bw, kernel='bisquare', fixed=False)
    gwr_results = gwr_model.fit()
    
    # Extract t-values (Index 1=Ethnic, Index 2=Illness)
    gdf_clean['gwr_ethnic'] = gwr_results.params[:, 1]
    gdf_clean['gwr_lt_illness'] = gwr_results.params[:, 2]
    gdf_clean['gwr_t_ethnic'] = gwr_results.tvalues[:, 1]
    gdf_clean['gwr_t_illness'] = gwr_results.tvalues[:, 2]
    
    # Plotting Function
    def plot_significant_map(column, t_value_col, title, filename):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot all units in grey first (insignificant background)
        gdf_clean.plot(ax=ax, color='lightgrey', edgecolor='white')
        
        # Plot significant units (|t| > 1.96)
        sig_mask = abs(gdf_clean[t_value_col]) > 1.96
        
        if sig_mask.sum() > 0:
            gdf_clean[sig_mask].plot(column=column, ax=ax, legend=True, 
                               legend_kwds={'title': title, 'loc': 'lower right'},
                               cmap='viridis', scheme='quantiles')
        
        plt.title(f'{title} (Signifikan t > 1.96)')
        plt.savefig(f'd:/tk1/figs/{filename}')
        plt.close()
        print(f"Saved {filename}")

    print("Generating maps...")
    plot_significant_map('gwr_ethnic', 'gwr_t_ethnic', 'Koefisien Etnis', 'covid_gwr_coef_ethnic_sig.png')
    plot_significant_map('gwr_lt_illness', 'gwr_t_illness', 'Koefisien Penyakit Jangka Panjang', 'covid_gwr_coef_lt_illness_sig.png')
    
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
