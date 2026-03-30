import os
import kagglehub
import pandas as pd
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from libpysal.weights import DistanceBand
from esda.moran import Moran
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def generate_latex_table(df, filename, caption, label):
    # Menyimpan dataframe menjadi format Tex murni untuk diload di dokumen utama
    latex_code = df.to_latex(index=True, escape=False, float_format="%.4f")
    # Menambahkan environment table otomatis
    full_code = "\\begin{table}[H]\n\\centering\n\\caption{" + caption + "}\n\\label{tab:" + label + "}\n"
    # Memperkecil tabel agar muat
    full_code += "\\footnotesize\n"
    full_code += latex_code
    full_code += "\\end{table}\n"
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
        f.write(full_code)

def main():
    print("="*50)
    print("  TAHAPAN 1 & 2: PENGUMPULAN & PEMBACAAN DATA")
    print("="*50)
    path = kagglehub.dataset_download("yenroyenro/harga-rumah-di-dki-jakarta-2024")
    
    data_file = None
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith('.geojson'):
                data_file = os.path.join(path, f)
                break
                
    if not data_file:
        print("[ERROR] File GeoJSON tidak ditemukan!")
        return
        
    df = gpd.read_file(data_file)
    print(f"[INFO] Memetakan variabel Y, X1, X2, X3, X5, X6, X7...")
    df = df.to_crs(epsg=32748)
    monas = gpd.GeoSeries(gpd.points_from_xy([106.827153], [-6.175392]), crs="EPSG:4326").to_crs(epsg=32748)
    
    df['X4'] = df.geometry.distance(monas.geometry[0]) / 1000  # meter ke km
    df['Y'] = df['Hrg_Mlr']
    df['X1'] = df['LB']
    df['X2'] = df['LT']
    df['X3'] = df['KT']
    df['X5'] = df['shortest_distance_school'] / 1000 
    df['X6'] = df['shortest_distance_health'] / 1000
    df['X7'] = df['jark_rd'] / 1000 if 'jark_rd' in df.columns else np.nan

    analisis_df = df[['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'geometry']].dropna()
    coords = np.array([(geom.x, geom.y) for geom in analisis_df.geometry])
    y = analisis_df['Y'].values.reshape(-1, 1)
    X = analisis_df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*50)
    print("  TAHAPAN 3: STATISTIK DESKRIPTIF & KORELASI")
    print("="*50)
    df_no_geom = analisis_df.drop(columns='geometry')
    # Descriptive Stats
    df_desc = df_no_geom.describe().T[['count', 'min', 'max', 'mean', 'std']]
    generate_latex_table(df_desc, "tabel_desc_stats.tex", "Statistik Deskriptif Variabel", "desc_stats")
    
    # Correlation Matrix
    df_corr = df_no_geom.corr()
    generate_latex_table(df_corr, "tabel_korelasi.tex", "Matriks Korelasi Variabel", "korelasi")
    
    # Heatmap Image
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriks Korelasi (Heatmap)")
    plt.savefig(os.path.join(out_dir, "peta_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("  TAHAPAN 4: ESTIMASI OLS, UJI BP & RESIDUAL")
    print("="*50)
    X_std = sm.add_constant(X)
    ols_model = sm.OLS(y, X_std).fit()
    
    ols_params_name = ['Intercept', 'Luas Bangunan (X1)', 'Luas Tanah (X2)', 'Jml Kamar (X3)', 'Jarak Pusat Kota (X4)', 'Jarak Sekolah (X5)', 'Jarak RS (X6)', 'Jarak Jalan Utama (X7)']
    df_ols = pd.DataFrame({
        "Koefisien": ols_model.params,
        "Std.Error": ols_model.bse,
        "t-Stat": ols_model.tvalues,
        "P-Value": ols_model.pvalues
    }, index=ols_params_name)
    generate_latex_table(df_ols, "tabel_ols.tex", "Hasil Estimasi Regresi Global (OLS)", "ols_result")
    
    # Breusch-Pagan Test
    bp_test = het_breuschpagan(ols_model.resid, X_std)
    df_bp = pd.DataFrame({
        "Statistik": ["Lagrange Multiplier", "LM p-value", "F-Statistic", "F p-value"],
        "Nilai": bp_test
    }).set_index("Statistik")
    generate_latex_table(df_bp, "tabel_bp.tex", "Hasil Uji Breusch-Pagan (Heteroskedastisitas)", "bp_test")
    
    # Map OLS Residuals
    dataset_gwr = analisis_df.copy()
    dataset_gwr['OLS_Residuals'] = ols_model.resid
    fig, ax = plt.subplots(figsize=(10, 8))
    dataset_gwr.plot(column='OLS_Residuals', cmap='coolwarm', legend=True, ax=ax, markersize=15, alpha=0.8)
    ax.set_title("Peta Error / Residual OLS Secara Lokal Spasial")
    ax.set_axis_off()
    plt.savefig(os.path.join(out_dir, "map_ols_residuals.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("  TAHAPAN 5: UJI MORAN'S I (AUTOKORELASI)")
    print("="*50)
    w = DistanceBand.from_dataframe(analisis_df, threshold=3000, binary=True, alpha=-1.0) 
    mi = Moran(analisis_df['Y'], w)
    df_moran = pd.DataFrame({
        "Statistik": ["Moran's I (Variabel Y)", "P-Value"],
        "Nilai": [mi.I, mi.p_sim]
    }).set_index("Statistik")
    generate_latex_table(df_moran, "tabel_moran.tex", "Hasil Uji Autokorelasi Spasial Moran's I", "moran_result")

    print("\n" + "="*50)
    print("  TAHAPAN 6: KALKULASI GWR (BISQUARE vs GAUSSIAN)")
    print("="*50)
    
    # Bisquare Mode
    print("[INFO] Mencari bobot Bandwidth terbaik untuk Bisquare...")
    selector_bsq = Sel_BW(coords, y, X, kernel='bisquare')
    bw_bsq = selector_bsq.search()
    gwr_bsq = GWR(coords, y, X, bw_bsq, kernel='bisquare').fit()
    
    # Gaussian Mode
    print("[INFO] Mencari bobot Bandwidth terbaik untuk Gaussian...")
    selector_gaus = Sel_BW(coords, y, X, kernel='gaussian')
    bw_gaus = selector_gaus.search()
    gwr_gaus = GWR(coords, y, X, bw_gaus, kernel='gaussian').fit()
    
    # Model Comparison
    df_compare = pd.DataFrame({
        "Mode Kinerja": ["Bandwidth", "R-Squared", "AIC", "AICc", "Log-Likelihood"],
        "OLS Global": ["-", f"{ols_model.rsquared:.4f}", f"{ols_model.aic:.4f}", "-", f"{ols_model.llf:.4f}"],
        "GWR Bisquare": [f"{bw_bsq:.4f}", f"{gwr_bsq.R2:.4f}", f"{gwr_bsq.aic:.4f}", f"{gwr_bsq.aicc:.4f}", f"{gwr_bsq.llf:.4f}"],
        "GWR Gaussian": [f"{bw_gaus:.4f}", f"{gwr_gaus.R2:.4f}", f"{gwr_gaus.aic:.4f}", f"{gwr_gaus.aicc:.4f}", f"{gwr_gaus.llf:.4f}"]
    }).set_index("Mode Kinerja")
    generate_latex_table(df_compare, "tabel_model_comparison.tex", "Perbandingan Kinerja Model OLS dan GWR (Bisquare vs Gaussian)", "model_comparison")

    # Deskriptif Koefisien GWR Bisquare (sebagai model utama pilihan)
    local_params = gwr_bsq.params
    df_gwr_coef = pd.DataFrame(local_params, columns=ols_params_name)
    df_gwr_coef_desc = df_gwr_coef.describe().T[['min', 'max', 'mean', 'std']]
    generate_latex_table(df_gwr_coef_desc, "tabel_gwr_coef.tex", "Statistik Deskriptif Koefisien Lokal GWR (Bisquare)", "gwr_coef")
    
    print("\n" + "="*50)
    print("  TAHAPAN 7: VISUALISASI KE PETA PENDUKUNG")
    print("="*50)
    
    dataset_gwr['Local_R2'] = gwr_bsq.localR2
    for i, col in enumerate(ols_params_name):
        dataset_gwr[f'Coef_{col.split(" ")[0]}'] = local_params[:, i]
    
    print("[INFO] Mengekspor dataset lengkap ke hasil_gwr_jakarta.geojson...")
    dataset_gwr.to_crs(epsg=4326).to_file(os.path.join(out_dir, "hasil_gwr_jakarta.geojson"), driver="GeoJSON")

    # Plot Local R2
    fig, ax = plt.subplots(figsize=(10, 8))
    dataset_gwr.plot(column='Local_R2', cmap='viridis', legend=True, ax=ax, markersize=15, alpha=0.8)
    ax.set_title("Sebaran Nilai Local R-Squared (GWR Bisquare)")
    ax.set_axis_off()
    plt.savefig(os.path.join(out_dir, "peta_Local_R2.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Coef Variables
    print("[INFO] Merender gambar persebaran setiap koefisien spasial (X1-X7)...")
    for col in ols_params_name:
        sanitized_col = col.split(" ")[0].replace("(", "").replace(")", "")
        col_name = f'Coef_{sanitized_col}'
        fig, ax = plt.subplots(figsize=(10, 8))
        dataset_gwr.plot(column=col_name, cmap='bwr', legend=True, ax=ax, markersize=15, alpha=0.8)
        ax.set_title(f"Distribusi Koefisien Lokal: {col}")
        ax.set_axis_off()
        plt.savefig(os.path.join(out_dir, f"peta_koefisien_{sanitized_col}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    print("[SELESAI] Seluruh metrik LaTeX Jurnal dan Peta Gambar berhasil digenerate di folder 5_Our!")

if __name__ == "__main__":
    main()
