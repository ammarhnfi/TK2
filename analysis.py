
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from libpysal.weights import Queen
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
import os

warnings.filterwarnings('ignore')

def save_latex_table(df, filename):
    """Saves a DataFrame to a simple LaTeX table file."""
    latex_code = df.to_latex(index=True, float_format="{:.4f}".format)
    
    # Manual booktabs replacement for older pandas versions
    # Replace the top hline
    latex_code = latex_code.replace('\\toprule', '\\hline') # standardizing just in case
    latex_code = latex_code.replace('\\midrule', '\\hline')
    latex_code = latex_code.replace('\\bottomrule', '\\hline')
    
    # Now replace them sequentially. 
    # This assumes standard pandas output with 3 horizontal lines.
    # It loops through lines.
    lines = latex_code.split('\n')
    new_lines = []
    hlines_seen = 0
    total_hlines = latex_code.count('\\hline')
    
    for line in lines:
        if '\\hline' in line:
            hlines_seen += 1
            if hlines_seen == 1:
                line = line.replace('\\hline', '\\toprule')
            elif hlines_seen == 2:
                line = line.replace('\\hline', '\\midrule')
            elif hlines_seen == total_hlines and total_hlines >= 3:
                line = line.replace('\\hline', '\\bottomrule')
        new_lines.append(line)
        
    latex_code = '\n'.join(new_lines)

    with open(f'tables/{filename}', 'w') as f:
        f.write(latex_code)

def run_covid_analysis():
    print("\n--- Running UK Covid-19 Analysis ---")
    
    try:
        # Load Data
        print("Loading data...")
        gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
        
        # Calculate rates and proportions if not already present or just to be safe/explicit
        # Based on reference:
        # covid19_r = (X2020.04.14 / Rsdnt) * 100000
        # ethnic = (Mixed + Indin + Pkstn + Bngld + Chins + Oth_A + Black + Othr_t) / Rsdnt
        # lt_illness = Lng__ / Rsdnt
        
        # Ensure we have the right columns. The dataset likely already has them or raw cols.
        # Let's assume standard names based on previous steps or recalculate.
        # Check if columns exist
        if 'covid19_r' not in gdf.columns:
             gdf['covid19_r'] = (gdf['X2020.04.14'] / gdf['Residents']) * 100000
        if 'ethnic' not in gdf.columns:
            # Summing ethnic columns: Mixed, Indian, Pakistani, Bangladeshi, Chinese, Other_Asian, Black, Other_ethnicity
            gdf['ethnic'] = (gdf['Mixed'] + gdf['Indian'] + gdf['Pakistani'] + gdf['Bangladeshi'] + 
                           gdf['Chinese'] + gdf['Other_Asian'] + gdf['Black'] + gdf['Other_ethnicity']) / gdf['Residents']
        if 'lt_illness' not in gdf.columns:
            gdf['lt_illness'] = gdf['Long_term_ill'] / gdf['Residents']
            
        # Clean Data (Fill NaNs)
        gdf = gdf.fillna(0)
        
        # Select key variables
        gdf_clean = gdf[['covid19_r', 'ethnic', 'lt_illness', 'geometry']].copy()
        
        # 0. Generate Descriptive Stats
        print("Generating Descriptive Statistics...")
        desc_stats = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].describe().T
        # Add skewness and kurtosis
        desc_stats['skew'] = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].skew()
        desc_stats['kurtosis'] = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].kurtosis()
        
        # Escape % in column names (25%, 50%, 75%)
        desc_stats.columns = [str(c).replace('%', '\\%') for c in desc_stats.columns]
        # Escape _ in index names (covid19_r, lt_illness)
        desc_stats.index = [str(i).replace('_', '\\_') for i in desc_stats.index]
        
        save_latex_table(desc_stats, 'desc_stats.tex')
        
        # Boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=gdf_clean[['covid19_r', 'ethnic', 'lt_illness']])
        plt.title('Boxplot Variabel')
        plt.savefig('figs/boxplot.png')
        plt.close()
        
        # Maps for Independent/Dependent Vars
        for var in ['covid19_r', 'ethnic', 'lt_illness']:
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_clean.plot(column=var, ax=ax, legend=True, cmap='OrRd')
            ax.set_title(f'Peta Sebaran: {var}')
            plt.savefig(f'figs/map_{var}.png')
            plt.close()

        # --- 1. Correlation Matrix ---
        print("Generating Correlation Matrix...")
        corr_matrix = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.savefig('figs/correlation_matrix.png')
        plt.close()

        # --- 2. OLS Analysis ---
        print("Performing OLS Analysis...")
        X = gdf_clean[['ethnic', 'lt_illness']]
        y = gdf_clean['covid19_r']
        
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(y, X_ols).fit()
        
        # Export OLS Result Summary manually to LaTeX for better control
        ols_latex = ols_model.summary().as_latex()
        # Remove center environment to allow better control in LaTeX
        ols_latex = ols_latex.replace(r'\begin{center}', '').replace(r'\end{center}', '')
        with open('tables/ols_summary.tex', 'w') as f:
            f.write(ols_latex)
            
        # Breusch-Pagan Test
        print("Running Breusch-Pagan Test...")
        bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        bp_results = pd.DataFrame(dict(zip(labels, bp_test)), index=['Value']).T
        save_latex_table(bp_results, 'bp_test.tex')
        
        # Map OLS Residuals
        print("Mapping OLS Residuals...")
        try:
            gdf_clean['ols_resid'] = ols_model.resid
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='ols_resid', ax=ax, legend=True,
                         legend_kwds={'orientation': "horizontal"},
                         cmap='RdBu', scheme='quantiles')
            plt.title('Peta Residual OLS')
            plt.savefig('figs/map_ols_residuals.png')
            plt.close()
        except Exception as e:
            print(f"Error mapping OLS residuals: {e}")

        # --- 3. GWR Analysis ---
        print("Selecting optimal bandwidth for GWR...")
        # Prepare coordinates
        coords = list(zip(gdf_clean.geometry.centroid.x, gdf_clean.geometry.centroid.y))
        
        # Select optimal bandwidth (Adaptive)
        # Using bisquare kernel for compact support as per reference suggestions for GWR
        gwr_selector = Sel_BW(coords, y.values.reshape(-1, 1), X.values, kernel='bisquare', fixed=False)
        bw = gwr_selector.search(criterion='AICc')
        print(f"Optimal bandwidth (nearest neighbors): {bw}")
        
        print("Fitting GWR model...")
        gwr_model = GWR(coords, y.values.reshape(-1, 1), X.values, bw, kernel='bisquare', fixed=False)
        gwr_results = gwr_model.fit()
        
        # Export GWR Summary
        gwr_summary = pd.DataFrame({
            'Metric': ['Bandwidth', 'AICc', 'R2', 'Adj. R2'],
            'Value': [bw, gwr_results.aicc, gwr_results.R2, gwr_results.adj_R2]
        }).set_index('Metric')
        save_latex_table(gwr_summary, 'gwr_summary.tex')

        # Add results to GeoDataFrame
        gdf_clean['gwr_intercept'] = gwr_results.params[:, 0]
        gdf_clean['gwr_ethnic'] = gwr_results.params[:, 1]
        gdf_clean['gwr_lt_illness'] = gwr_results.params[:, 2]
        gdf_clean['gwr_r2'] = gwr_results.localR2
        gdf_clean['gwr_t_ethnic'] = gwr_results.tvalues[:, 1]
        gdf_clean['gwr_t_illness'] = gwr_results.tvalues[:, 2]
        with open('gwr_attrs.txt', 'w') as f:
            f.write(str(dir(gwr_results)))
        print("GWR attributes written to gwr_attrs.txt")
        
        try:
            # Index 2 identified as Condition Number from debug analysis (values ~15-23)
            cn = gwr_results.local_collinearity()[2]
            if hasattr(cn, 'shape') and len(cn.shape) > 1:
                gdf_clean['condition_number'] = cn.flatten()
            else:
                gdf_clean['condition_number'] = cn
            
            print("Local collinearity calculated.")
            
        except Exception as e:
            print(f"Error calculating local collinearity: {e}")

        # --- 4. Visualizations ---
        print("Generating GWR visualizations...")
        
        # Generate Map Function with Significance Masking
        def plot_significant_map(column, t_value_col, title, filename):
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                # Plot all units in grey first (insignificant background)
                gdf_clean.plot(ax=ax, color='lightgrey', edgecolor='white')
                
                # Plot significant units (|t| > 1.96)
                sig_mask = abs(gdf_clean[t_value_col]) > 1.96
                if sig_mask.sum() > 0:
                    gdf_clean[sig_mask].plot(column=column, ax=ax, legend=True, 
                                       legend_kwds={'orientation': "horizontal"},
                                       cmap='viridis', scheme='quantiles')
                
                plt.title(f'{title} (Signifikan t > 1.96)')
                plt.savefig(f'figs/{filename}')
                plt.close()
            except Exception as e:
                print(f"Error plotting {filename}: {e}")

        # Coefficients with Significance
        plot_significant_map('gwr_ethnic', 'gwr_t_ethnic', 'Koefisien Etnis', 'covid_gwr_coef_ethnic_sig.png')
        plot_significant_map('gwr_lt_illness', 'gwr_t_illness', 'Koefisien Penyakit Jangka Panjang', 'covid_gwr_coef_lt_illness_sig.png')
        
        # Also save the t-values maps themselves for reference
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_t_ethnic', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3)
            plt.title('t-values Etnis')
            plt.savefig('figs/gwr_tval_ethnic.png')
            plt.close()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_t_illness', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3)
            plt.title('t-values Penyakit')
            plt.savefig('figs/gwr_tval_illness.png')
            plt.close()
        except Exception as e:
             print(f"Error plotting t-values: {e}")
        
        # Local R2
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_r2', ax=ax, legend=True, cmap='plasma')
            plt.title('Local R2')
            plt.savefig('figs/covid_gwr_r2.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting local R2: {e}")

        # Local Collinearity (Condition Number)
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='condition_number', ax=ax, legend=True, cmap='Reds', scheme='quantiles')
            plt.title('Local Condition Number (Multikolinearitas)')
            plt.savefig('figs/map_condition_number.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting condition number: {e}")

        print("Covid analysis completed successfully.")
    
    except Exception as e:
        print(f"Error in Covid analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists('figs'):
        os.makedirs('figs')
    if not os.path.exists('tables'):
        os.makedirs('tables')
        
    run_covid_analysis()
