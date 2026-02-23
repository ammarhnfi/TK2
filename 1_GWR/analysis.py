
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
import warnings
import os
from scipy.stats import t

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
        if 'imd' not in gdf.columns:
            # Check for IMD column (name might vary in shp vs gpkg, checking Untitled106)
            # Untitled106 used: gdf['IMD...Average.score']
            # Let's try to find it dynamically or hardcode if we trust the gpkg
            potential_imd = [c for c in gdf.columns if 'IMD' in c and 'Average' in c]
            if potential_imd:
                gdf['imd'] = gdf[potential_imd[0]]
            else:
                # Fallback or error
                print("Warning: IMD column not found via fuzzy match, trying exact 'IMD...Average.score'")
                if 'IMD...Average.score' in gdf.columns:
                    gdf['imd'] = gdf['IMD...Average.score']
                else:
                     raise ValueError("Column 'IMD...Average.score' not found in dataset.")


        if 'lt_illness' not in gdf.columns:
            gdf['lt_illness'] = gdf['Long_term_ill'] / gdf['Residents']

        if 'crowded' not in gdf.columns:
             gdf['crowded'] = gdf['Crowded_housing'] / gdf['Households']
            
        # Clean Data (Fill NaNs)
        gdf = gdf.fillna(0)
        
        # Select key variables
        gdf_clean = gdf[['covid19_r', 'imd', 'lt_illness', 'crowded', 'geometry']].copy()
        
        # Save cleaned data to CSV for report
        # (Old 4-column CSV export removed to prefer the 7-column one below)
        
        # 0. Generate Descriptive Stats
        print("Generating Descriptive Statistics...")
        vars_to_analyze = ['covid19_r', 'imd', 'lt_illness', 'crowded']
        desc_stats = gdf_clean[vars_to_analyze].describe()
        # Add skewness and kurtosis as rows
        desc_stats.loc['skewness'] = gdf_clean[vars_to_analyze].skew()
        desc_stats.loc['kurtosis'] = gdf_clean[vars_to_analyze].kurtosis()
        
        # Escape _ in column names (covid19_r, lt_illness)
        desc_stats.columns = [str(c).replace('_', '\\_') for c in desc_stats.columns]
        # Escape % in index names (25%, 50%, 75%)
        desc_stats.index = [str(i).replace('%', '\\%') for i in desc_stats.index]
        
        save_latex_table(desc_stats, 'desc_stats.tex')
        
        # Prepare Combined Data Sample (Location + Variables)
        # Identify Name Column
        name_col = 'ctyua19nm'
        if name_col not in gdf.columns:
             name_col = [c for c in gdf.columns if 'name' in c.lower() or 'lad' in c.lower()][0]
        
        # Identify Lat/Lon Columns
        if 'lat' in gdf.columns and 'long' in gdf.columns:
             lat_col = 'lat'
             lon_col = 'long'
        else:
             lat_col = 'Latitude'
             lon_col = 'Longitude'
             gdf[lon_col] = gdf.geometry.centroid.x
             gdf[lat_col] = gdf.geometry.centroid.y
             
        # Create Data Analysis CSV with Location Info
        # Columns: Name, Long, Lat, Y, X1, X2, X3...
        cols_to_export = [name_col, lon_col, lat_col, 'covid19_r', 'imd', 'lt_illness', 'crowded']
        gdf_export = gdf[cols_to_export].copy()
        
        try:
            gdf_export.to_csv('data_analysis.csv', index=False)
            print("Exported data_analysis.csv with location info.")
        except PermissionError:
             print("Warning: Could not write to data_analysis.csv (File locked).")
        except Exception as e:
             print(f"Error exporting CSV: {e}")

        # Generate Sample Table (Merged)
        data_sample = gdf_export.head(10).copy()
        data_sample.columns = ['Region Name', 'Longitude', 'Latitude', 'Covid Rate', 'IMD', 'Illness', 'Crowded']
        save_latex_table(data_sample, 'data_sample.tex')
        print("Generated data_sample.tex with 7 columns.")
             
        # --- 0.5 Location Map & Data Table ---
        print("Generating Reference Map and Location Table...")
        
        # 2. Reference Map (Boundaries + Labels + Axis)
        fig, ax = plt.subplots(figsize=(12, 12))
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
        
        # Add labels using the specified columns
        for idx, row in gdf.iterrows():
            # Only annotate if area is large enough to avoid clutter or just plot all
            plt.annotate(text=row[name_col], xy=(row[lon_col], row[lat_col]),
                         horizontalalignment='center', fontsize=6, color='darkblue')
        
        plt.title('Peta Wilayah Studi dengan Batas Administrasi')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, which='major', linestyle='--', alpha=0.5)
        plt.savefig('figs/map_boundaries.png')
        plt.close()
        
        # Boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=gdf_clean[vars_to_analyze])
        plt.title('Boxplot Variabel')
        plt.savefig('figs/boxplot.png')
        plt.close()

        # Pairplot
        print("Generating Pairplot...")
        sns.pairplot(gdf_clean[vars_to_analyze])
        plt.savefig('figs/pairplot.png')
        plt.close()

        # Heatmap (Correlation Matrix)
        print("Generating Correlation Heatmap...")
        plt.figure(figsize=(8, 6))
        corr_matrix = gdf_clean[vars_to_analyze].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Korelasi Antar Variabel')
        plt.savefig('figs/heatmap.png')
        plt.close()
        
        # Maps for Independent/Dependent Vars
        for var in vars_to_analyze:
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_clean.plot(column=var, ax=ax, legend=True, cmap='OrRd')
            ax.set_title(f'Peta Sebaran: {var}')
            plt.savefig(f'figs/map_{var}.png')
            plt.close()

        # --- 1. Correlation Matrix ---
        print("Generating Correlation Matrix...")
        corr_matrix = gdf_clean[vars_to_analyze].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.savefig('figs/correlation_matrix.png')
        plt.close()

        # --- 2. Global Regression (OLS) ---
        print("Running Global Regression (OLS)...")
        
        # STANDARDIZATION (Z-Score)
        # Standardize variables to ensure comparable coefficients
        # Note: Descriptive stats above used original values, which is correct.
        print("Standardizing variables (Z-score)...")
        gdf_standardized = gdf_clean.copy()
        for col in vars_to_analyze:
            gdf_standardized[col] = (gdf_clean[col] - gdf_clean[col].mean()) / gdf_clean[col].std()
        
        # Update y and X to use standardized data
        y = gdf_standardized['covid19_r']
        X = gdf_standardized[['imd', 'lt_illness', 'crowded']]
        
        # Add constant for OLS
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
            # Add standardized residuals to ORIGINAL gdf_clean for mapping (keeps geometry)
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

        # --- 3. GWR Analysis (Comparing Kernels) ---
        print("Selecting optimal bandwidths and fitting GWR models...")
        coords = list(zip(gdf_clean.geometry.centroid.x, gdf_clean.geometry.centroid.y))
        
        # Function to run GWR and return results
        def run_gwr(kernel_name):
            print(f"Running GWR with {kernel_name} kernel...")
            # Disable multiprocessing (multi=False) to avoid Windows issues
            selector = Sel_BW(coords, y.values.reshape(-1, 1), X.values, kernel=kernel_name, fixed=False, multi=False)
            bw = selector.search(criterion='AICc')
            print(f"Optimal bandwidth ({kernel_name}): {bw}")
            model = GWR(coords, y.values.reshape(-1, 1), X.values, bw, kernel=kernel_name, fixed=False)
            results = model.fit()
            return bw, results

        # 1. Adaptive Bisquare
        bw_bisquare, gwr_bisquare = run_gwr('bisquare')
        
        # 2. Adaptive Gaussian
        bw_gaussian, gwr_gaussian = run_gwr('gaussian')

        # Compare Models
        comparison_data = {
            'Model': ['OLS', 'GWR Bisquare', 'GWR Gaussian'],
            'AICc': [ols_model.aic, gwr_bisquare.aicc, gwr_gaussian.aicc],
            'R2': [ols_model.rsquared, gwr_bisquare.R2, gwr_gaussian.R2],
            'Adj. R2': [ols_model.rsquared_adj, gwr_bisquare.adj_R2, gwr_gaussian.adj_R2]
        }
        comparison_df = pd.DataFrame(comparison_data)
        save_latex_table(comparison_df.set_index('Model'), 'model_comparison.tex')
        
        # Choose Best Model (Lower AICc)
        if gwr_bisquare.aicc < gwr_gaussian.aicc:
            print("Bisquare is better. Using Bisquare for detailed maps.")
            best_gwr = gwr_bisquare
            best_kernel = 'Bisquare'
        else:
            print("Gaussian is better. Using Gaussian for detailed maps.")
            best_gwr = gwr_gaussian
            best_kernel = 'Gaussian'

        # Export Best GWR Summary (for backward compatibility if needed, but we use comparison table now)
        gwr_summary = pd.DataFrame({
            'Metric': ['Bandwidth', 'AICc', 'R2', 'Adj. R2'],
            'Value': [bw_bisquare, best_gwr.aicc, best_gwr.R2, best_gwr.adj_R2]
        }).set_index('Metric')
        save_latex_table(gwr_summary, 'gwr_summary.tex')

        # Save Residuals for Analysis
        gdf_clean['resid_ols'] = ols_model.resid
        gdf_clean['resid_gwr_bisquare'] = gwr_bisquare.resid_response
        gdf_clean['resid_gwr_gaussian'] = gwr_gaussian.resid_response
        
        # Use Best Model attributes for coefficient mapping
        gwr_results = best_gwr 

        # Add results to GeoDataFrame
        gdf_clean['gwr_intercept'] = gwr_results.params[:, 0]
        gdf_clean['gwr_imd'] = gwr_results.params[:, 1]
        gdf_clean['gwr_lt_illness'] = gwr_results.params[:, 2]
        gdf_clean['gwr_crowded'] = gwr_results.params[:, 3]
        
        gdf_clean['gwr_r2'] = gwr_results.localR2
        
        gdf_clean['gwr_t_imd'] = gwr_results.tvalues[:, 1]
        gdf_clean['gwr_t_illness'] = gwr_results.tvalues[:, 2]
        gdf_clean['gwr_t_crowded'] = gwr_results.tvalues[:, 3]
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
        
        # Plot Residual Maps (Comparison)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # OLS
        gdf_clean.plot(column='resid_ols', ax=axes[0], legend=True, 
                     legend_kwds={'orientation': "horizontal", 'label': "Residuals"},
                     cmap='RdBu')
        axes[0].set_title(f'OLS Residuals\nAICc: {ols_model.aic:.2f}')
        axes[0].axis('off')

        # GWR Bisquare
        gdf_clean.plot(column='resid_gwr_bisquare', ax=axes[1], legend=True,
                     legend_kwds={'orientation': "horizontal", 'label': "Residuals"},
                     cmap='RdBu')
        axes[1].set_title(f'GWR Bisquare Residuals\nAICc: {gwr_bisquare.aicc:.2f}')
        axes[1].axis('off')

        # GWR Gaussian
        gdf_clean.plot(column='resid_gwr_gaussian', ax=axes[2], legend=True,
                     legend_kwds={'orientation': "horizontal", 'label': "Residuals"},
                     cmap='RdBu')
        axes[2].set_title(f'GWR Gaussian Residuals\nAICc: {gwr_gaussian.aicc:.2f}')
        axes[2].axis('off')

        plt.suptitle('Perbandingan Peta Residual: OLS vs GWR Bisquare vs GWR Gaussian', fontsize=16)
        plt.savefig('figs/residual_comparison.png')
        plt.close()
        
        # Generate Separate Maps for Coef and P-value
        def plot_coef_and_pval(coef_col, t_col, title_base, filename_base):
            try:
                # 1. Coefficient Map
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                gdf_clean.plot(column=coef_col, ax=ax, legend=True, 
                               cmap='coolwarm', scheme='quantiles', k=5)
                plt.title(f'Koefisien {title_base}')
                plt.savefig(f'figs/{filename_base}_coef.png')
                plt.close()

                # 2. P-value Map
                # Calculate p-values from t-values (two-tailed)
                # df = n - k - 1? approx large n use normal or t with localized df
                # convenient way: 2 * (1 - t.cdf(abs(t), df))
                # For simplicity and given GWR complexity, use standard normal approx if df large, 
                # but let's use t-dist with n-k degrees of freedom roughly. 
                # GWR effective parameters vary. 
                # Let's use |t| > 1.96 as standard threshold for visualization (p<0.05).
                # To map the actual p-value:
                # p_values = t.sf(abs(gdf_clean[t_col]), gwr_results.df_resid) * 2 
                # But gwr_results might have effective df. 
                # Let's map the p-value directly if we can, or just use t-values visually.
                # User asked for "p-value".
                
                # Let's compute p-values using the flexible t distribution from statsmodels or scipy
                # We need Degrees of Freedom. gwr_results.df_resid is global? 
                # Local t-values are computed using local standard errors.
                # Let's trust the t-values and map probability.
                # P = 2 * (1 - CDF(|t|)) approx by Normal if n is large.
                # England UA n=149.
                
                p_values = 2 * (1 - t.cdf(abs(gdf_clean[t_col]), df=gwr_results.df_resid))
                gdf_clean[f'pval_{coef_col}'] = p_values
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                # Map p-values. Usually focus on < 0.05, 0.01 etc.
                # Let's use a custom classification or continuous.
                # Reverse cmap so small p-value (significant) is dark/highlighted?
                # Or just 'RdYlGn_r' where Red is low p-value (significant)? 
                # 'viridis_r' is good. 0 is bright yellow, 1 is purple. 
                # Actually typically "Significance" maps categorize: <0.01, <0.05, <0.1, Not Sig.
                # But user just said "satu untuk p-value nya".
                gdf_clean.plot(column=f'pval_{coef_col}', ax=ax, legend=True,
                               cmap='viridis_r', legend_kwds={'label': "P-Value"})
                plt.title(f'P-Value {title_base}')
                plt.savefig(f'figs/{filename_base}_pval.png')
                plt.close()

            except Exception as e:
                print(f"Error plotting {filename_base}: {e}")

        # Coefficients with Significance
        plot_coef_and_pval('gwr_imd', 'gwr_t_imd', 'IMD', 'covid_gwr_imd')
        plot_coef_and_pval('gwr_lt_illness', 'gwr_t_illness', 'Penyakit Jangka Panjang', 'covid_gwr_illness')
        plot_coef_and_pval('gwr_crowded', 'gwr_t_crowded', 'Kepadatan Hunian', 'covid_gwr_crowded')
        
        # Boxplot of GWR Coefficients (Betas)
        print("Generating GWR Coefficient Boxplots...")
        try:
             plt.figure(figsize=(10, 6))
             # Ensure these columns exist
             if all(col in gdf_clean.columns for col in ['gwr_intercept', 'gwr_imd', 'gwr_lt_illness', 'gwr_crowded']):
                 beta_cols = ['gwr_intercept', 'gwr_imd', 'gwr_lt_illness', 'gwr_crowded']
                 sns.boxplot(data=gdf_clean[beta_cols])
                 plt.title('Distribusi Koefisien Lokal (Beta) GWR')
                 plt.ylabel('Nilai Koefisien')
                 plt.savefig('figs/gwr_beta_boxplot.png')
                 plt.close()
             else:
                 print("Error: GWR beta columns not found for boxplot.")
        except Exception as e:
            print(f"Error plotting boxplot: {e}")

        # Also save the t-values maps themselves for reference
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_t_imd', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3)
            plt.title('t-values IMD')
            plt.savefig('figs/gwr_tval_imd.png')
            plt.close()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_t_illness', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3)
            plt.title('t-values Penyakit')
            plt.savefig('figs/gwr_tval_illness.png')
            plt.close()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf_clean.plot(column='gwr_t_crowded', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3)
            plt.title('t-values Crowded')
            plt.savefig('figs/gwr_tval_crowded.png')
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
