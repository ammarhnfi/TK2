
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import os
import sys

# Add current dir to path to import analysis
sys.path.append(os.getcwd())
try:
    from analysis import save_latex_table
except ImportError:
    # If fails, define it inline to ensure consistency with the fix I made
    def save_latex_table(df, filename):
        latex_code = df.to_latex(index=True, float_format="{:.4f}".format)
        latex_code = latex_code.replace('\\toprule', '\\hline')
        latex_code = latex_code.replace('\\midrule', '\\hline')
        latex_code = latex_code.replace('\\bottomrule', '\\hline')
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

def regen():
    print("Loading data...")
    gdf = gpd.read_file('d:/tk1/san_ref/data/assignment_2_covid/covid19_eng.gpkg')
    
    if 'covid19_r' not in gdf.columns:
            gdf['covid19_r'] = (gdf['X2020.04.14'] / gdf['Residents']) * 100000
    if 'ethnic' not in gdf.columns:
        gdf['ethnic'] = (gdf['Mixed'] + gdf['Indian'] + gdf['Pakistani'] + gdf['Bangladeshi'] + 
                        gdf['Chinese'] + gdf['Other_Asian'] + gdf['Black'] + gdf['Other_ethnicity']) / gdf['Residents']
    if 'lt_illness' not in gdf.columns:
        gdf['lt_illness'] = gdf['Long_term_ill'] / gdf['Residents']
        
    gdf = gdf.fillna(0)
    gdf_clean = gdf[['covid19_r', 'ethnic', 'lt_illness', 'geometry']].copy()
    
    print("Generating desc_stats.tex...")
    desc_stats = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].describe().T
    desc_stats['skew'] = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].skew()
    desc_stats['kurtosis'] = gdf_clean[['covid19_r', 'ethnic', 'lt_illness']].kurtosis()
    
    # Escape % in column names
    desc_stats.columns = [str(c).replace('%', '\\%') for c in desc_stats.columns]
    
    save_latex_table(desc_stats, 'desc_stats.tex')
    
    print("Generating OLS stats...")
    X = gdf_clean[['ethnic', 'lt_illness']]
    y = gdf_clean['covid19_r']
    X_ols = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols).fit()
    
    # bp_test
    bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_results = pd.DataFrame(dict(zip(labels, bp_test)), index=['Value']).T
    save_latex_table(bp_results, 'bp_test.tex')
    
    # ols_summary 
    ols_latex = ols_model.summary().as_latex()
    ols_latex = ols_latex.replace(r'\begin{center}', '').replace(r'\end{center}', '')
    with open('tables/ols_summary.tex', 'w') as f:
        f.write(ols_latex)

if __name__ == "__main__":
    if not os.path.exists('tables'):
        os.makedirs('tables')
    regen()
