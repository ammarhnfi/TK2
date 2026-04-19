"""
generate_desc_stats_pdf.py
Membuat PDF Statistika Deskriptif Detail dari data harga rumah DKI Jakarta 2024
"""

import json
import math
import subprocess
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─────────────────────────────────────────────
# 1. LOAD DATA dari GeoJSON
# ─────────────────────────────────────────────
GEOJSON_FILE = "rumah_27_8_25_clean.geojson"
with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
    geo = json.load(f)

features = geo["features"]

key_map = {
    "Y":  "Hrg_Mlr",
    "X1": "LB",                         # Luas Bangunan
    "X2": "LT",                         # Luas Tanah
    "X3": "KT",                         # Jumlah Kamar Tidur
    "X5": "shortest_distance_school",   # Jarak ke Sekolah (meter -> km)
    "X6": "shortest_distance_health",   # Jarak ke RS (meter -> km)
    "X7": "jark_rd",                    # Jarak ke Jalan Utama (meter -> km)
}

# Koordinat Monas (WGS84)
MONAS_LAT = -6.175392
MONAS_LON = 106.827153

def haversine_km(lat1, lon1, lat2, lon2):
    """Hitung jarak haversine dua titik dalam km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

raw = {v: [] for v in list(key_map.keys()) + ["X4"]}

METER_TO_KM = {"X5", "X6", "X7"}  # kolom yang nilainya dalam meter

for feat in features:
    props = feat["properties"]
    geom  = feat.get("geometry", {})
    coords_geom = geom.get("coordinates", None)

    for var, key in key_map.items():
        val = props.get(key)
        if val is not None:
            try:
                fval = float(val)
                if var in METER_TO_KM:
                    fval = fval / 1000.0
                raw[var].append(fval)
            except (ValueError, TypeError):
                pass

    # Hitung X4: jarak ke Monas dari koordinat geometry [lon, lat]
    if coords_geom and len(coords_geom) >= 2:
        try:
            lon, lat = float(coords_geom[0]), float(coords_geom[1])
            raw["X4"].append(haversine_km(lat, lon, MONAS_LAT, MONAS_LON))
        except Exception:
            pass

# ─────────────────────────────────────────────
# 2. FUNGSI STATISTIK
# ─────────────────────────────────────────────
def mean(d):    return sum(d) / len(d)
def variance(d):
    m = mean(d)
    return sum((x - m)**2 for x in d) / (len(d) - 1)
def std(d):     return math.sqrt(variance(d))
def median(d):
    s = sorted(d); n = len(s)
    return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]
def percentile(d, p):
    s = sorted(d); n = len(s)
    if n == 0: return 0.0
    if n == 1: return s[0]
    idx = (p/100)*(n-1)
    lo = max(0, min(int(idx), n-2))
    hi = lo + 1
    frac = idx - lo
    return s[lo] + frac*(s[hi]-s[lo])
def skewness(d):
    n = len(d); m = mean(d); s = std(d)
    if s == 0: return 0
    return (n/((n-1)*(n-2)))*sum(((x-m)/s)**3 for x in d)
def kurtosis(d):
    n = len(d); m = mean(d); s = std(d)
    if s == 0: return 0
    t1 = (n*(n+1))/((n-1)*(n-2)*(n-3))
    t2 = sum(((x-m)/s)**4 for x in d)
    t3 = 3*(n-1)**2/((n-2)*(n-3))
    return t1*t2 - t3
def cv(d):
    m = mean(d)
    return 0 if m==0 else (std(d)/abs(m))*100
def iqr(d): return percentile(d,75)-percentile(d,25)
def pearson(a, b):
    n = min(len(a),len(b)); ax=a[:n]; bx=b[:n]
    ma=mean(ax); mb=mean(bx)
    num=sum((ax[i]-ma)*(bx[i]-mb) for i in range(n))
    den=math.sqrt(sum((x-ma)**2 for x in ax)*sum((x-mb)**2 for x in bx))
    return 0 if den==0 else num/den

# ─────────────────────────────────────────────
# 3. HITUNG SEMUA STATISTIK & OUTLIER
# ─────────────────────────────────────────────
vars_list = ["Y","X1","X2","X3","X4","X5","X6","X7"]
stats = {}
df_for_plotting = pd.DataFrame(raw) # Convert to DataFrame for easier plotting and VIF

for var in vars_list:
    d = raw[var]
    q1 = percentile(d,25); q3 = percentile(d,75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    
    outliers = [x for x in d if x < lower_bound or x > upper_bound]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(d)) * 100 if len(d) > 0 else 0
    
    stats[var] = {
        "n": len(d), "min": min(d), "max": max(d),
        "Q1": q1, "Q3": q3, "median": median(d),
        "mean": mean(d), "std": std(d),
        "var": variance(d), "cv": cv(d), "iqr": iqr_val,
        "skew": skewness(d), "kurt": kurtosis(d),
        "range": max(d)-min(d),
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct
    }

# ─────────────────────────────────────────────
# 3b. PROSES VISUALISASI & VIF
# ─────────────────────────────────────────────
print("Menghasilkan Visualisasi (Boxplot & Pairplot)...")

# 1. Boxplot (Subplots because of scale difference)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
axes = axes.flatten()
for i, var in enumerate(vars_list):
    sns.boxplot(y=df_for_plotting[var], ax=axes[i], color='skyblue')
    axes[i].set_title(f"Boxplot {var}")
    axes[i].set_ylabel("")
plt.tight_layout()
plt.savefig("boxplot_deskriptif.png", dpi=300)
plt.close()

# 2. Pairplot
# We subset to important vars to keep it readable
pp = sns.pairplot(df_for_plotting, diag_kind='kde', plot_kws={'alpha':0.4, 's':10})
pp.fig.suptitle("Pairplot Sebaran dan Korelasi Antar Variabel", y=1.02)
plt.savefig("pairplot_deskriptif.png", dpi=300)
plt.close()

# 3. VIF Calculation (ONLY for X1-X7)
X_vif = df_for_plotting[["X1","X2","X3","X4","X5","X6","X7"]]
# Add constant for VIF logic usually required by statsmodels
X_vif_const = X_vif.copy()
X_vif_const['const'] = 1
vif_data = []
for i in range(len(X_vif.columns)):
    v = variance_inflation_factor(X_vif_const.values, i)
    vif_data.append({"Variable": X_vif.columns[i], "VIF": v})
vif_df = pd.DataFrame(vif_data)

# ─────────────────────────────────────────────
# 4. HELPERS FORMAT
# ─────────────────────────────────────────────
def fmt(x, dec=4):
    if abs(x) >= 10000:  return f"{x:,.0f}"
    elif abs(x) >= 1000: return f"{x:,.2f}"
    elif abs(x) >= 100:  return f"{x:.3f}"
    else:                return f"{x:.{dec}f}"

def skew_label(sk):
    if sk > 1.0:   return r"\textit{Right-Skewed Kuat}"
    if sk > 0.5:   return r"\textit{Right-Skewed Sedang}"
    if sk < -1.0:  return r"\textit{Left-Skewed Kuat}"
    if sk < -0.5:  return r"\textit{Left-Skewed Sedang}"
    return "Mendekati Simetris"

def kurt_label(ku):
    if ku > 3:   return "Leptokurtik"
    if ku < -1:  return "Platikurtik"
    return "Mesokurtik"

# ─────────────────────────────────────────────
# 5. DATA META PER VARIABEL
# ─────────────────────────────────────────────
unit_labels = {
    "Y":"Miliar Rp","X1":"m$^2$","X2":"m$^2$",
    "X3":"Kamar","X4":"km","X5":"km","X6":"km","X7":"km",
}

var_nicknames = {
    "Y": "Harga (Y)",
    "X1": "LB (X1)",
    "X2": "LT (X2)",
    "X3": "KT (X3)",
    "X4": "Monas (X4)",
    "X5": "Sekolah (X5)",
    "X6": "RS (X6)",
    "X7": "Jalan (X7)"
}
var_label_tex = {
    "Y": r"\texorpdfstring{$Y$}{Y} --- Harga Rumah (Miliar Rp)",
    "X1": r"\texorpdfstring{$X_1$ --- Luas Bangunan (m\textsuperscript{2})}{X1 --- Luas Bangunan (m2)}",
    "X2": r"\texorpdfstring{$X_2$ --- Luas Tanah (m\textsuperscript{2})}{X2 --- Luas Tanah (m2)}",
    "X3": r"\texorpdfstring{$X_3$}{X3} --- Jumlah Kamar",
    "X4": r"\texorpdfstring{$X_4$}{X4} --- Jarak ke Monas (km)",
    "X5": r"\texorpdfstring{$X_5$}{X5} --- Jarak ke Sekolah (km)",
    "X6": r"\texorpdfstring{$X_6$}{X6} --- Jarak ke RS (km)",
    "X7": r"\texorpdfstring{$X_7$}{X7} --- Jarak ke Jalan Utama (km)",
}
desc_interp = {
"Y": (
    r"Variabel dependen $Y$ (harga rumah) memiliki distribusi yang sangat menceng kanan "
    r"(\textit{right-skewed}) dengan nilai maksimum mencapai \textbf{Rp221 miliar}, "
    r"sementara median jauh di bawah rata-rata. "
    r"Hal ini menunjukkan adanya outlier ekstrem berupa properti mewah di Jakarta Selatan "
    r"dan Pusat. Koefisien variasi (CV) yang sangat tinggi menegaskan "
    r"heterogenitas kelas harga properti Jakarta."
),
"X1": (
    r"Luas bangunan ($X_1$) berkisar antara \textbf{36 m$^2$} (rumah tipe kecil) hingga "
    r"\textbf{4.850 m$^2$} (vila/mansion). Skewness positif tinggi mengindikasikan adanya "
    r"properti berukuran sangat besar yang membentuk ekor distribusi panjang."
),
"X2": (
    r"Luas kaveling tanah ($X_2$) menunjukkan dispersi tertinggi di antara variabel fisik, "
    r"dengan standar deviasi \textbf{448,8 m$^2$}. Fenomena ini lazim di Jakarta karena "
    r"variasi ekstrem antara rumah tapak kecil dan hunian besar di kawasan premium (Pondok Indah, Menteng)."
),
"X3": (
    r"Jumlah kamar ($X_3$) memiliki distribusi paling mendekati simetris di antara "
    r"variabel fisik, dengan median \textbf{3 kamar}. "
    r"Rentang 1--17 kamar mencerminkan segmentasi pasar dari unit studio hingga gedung residensial besar."
),
"X4": (
    r"Jarak ke Monas ($X_4$) sebagai proksi \textit{centrality} memiliki rata-rata \textbf{9,57 km}. "
    r"Distribusinya paling mendekati normal (skewness mendekati 0) di antara semua variabel, "
    r"mencerminkan persebaran properti yang cukup merata secara radial dari pusat kota."
),
"X5": (
    r"Jarak ke sekolah ($X_5$) memiliki nilai minimum 0 km dan maksimum \textbf{8,18 km}. "
    r"Rata-rata \textbf{1,44 km} menunjukkan bahwa sebagian besar properti relatif dekat "
    r"dengan fasilitas pendidikan. Skewness positif mengindikasikan "
    r"sebaran terkonsentrasi di jarak dekat."
),
"X6": (
    r"Jarak ke rumah sakit ($X_6$) berkorelasi sangat tinggi dengan $X_5$ ($r=0{,}747$), "
    r"menandakan pola klasterisasi fasilitas publik. Variabel ini memiliki pengaruh negatif "
    r"signifikan terhadap harga pada model OLS, mengindikasikan bahwa properti yang jauh "
    r"dari RS cenderung lebih murah."
),
"X7": (
    r"Jarak ke jalan utama ($X_7$) memiliki skala sangat kecil (rata-rata $\approx$0,016 km = 16 meter), "
    r"menunjukkan hampir seluruh properti berada sangat dekat dengan akses jalan utama. "
    r"CV ekstrem mengindikasikan distribusi sangat skewed dengan sebagian kecil properti "
    r"terisolasi jauh dari akses jalan."
),
}

# ─────────────────────────────────────────────
# 6. BANGUN BADAN TABEL-TABEL
# ─────────────────────────────────────────────

# Tabel 1: Summary statistics
tbl1_lines = []
for i, var in enumerate(vars_list):
    s = stats[var]
    row = (f"{var_nicknames[var]} & {int(s['n'])} & {fmt(s['min'])} & {fmt(s['Q1'])} & "
           f"{fmt(s['median'])} & {fmt(s['mean'])} & {fmt(s['Q3'])} & "
           f"{fmt(s['max'])} & {fmt(s['std'])} \\\\")
    if i % 2 == 0:
        tbl1_lines.append(r"\rowcolor{rowlight}")
    tbl1_lines.append(row)
tbl1_body = "\n".join(tbl1_lines)

# Tabel 2: Dispersi
tbl2_lines = []
for i, var in enumerate(vars_list):
    s = stats[var]
    row = (f"{var_nicknames[var]} & {fmt(s['var'],2)} & {fmt(s['cv'],2)}\\% & "
           f"{fmt(s['iqr'])} & {fmt(s['range'])} & "
           f"{fmt(s['skew'])} & {fmt(s['kurt'])} \\\\")
    if i % 2 == 0:
        tbl2_lines.append(r"\rowcolor{rowlight}")
    tbl2_lines.append(row)
tbl2_body = "\n".join(tbl2_lines)

# Tabel 3: Klasifikasi distribusi
tbl3_lines = []
for i, var in enumerate(vars_list):
    s = stats[var]
    row = (f"{var_nicknames[var]} & {fmt(s['skew'])} & {fmt(s['kurt'])} & "
           f"{skew_label(s['skew'])} & {kurt_label(s['kurt'])} \\\\")
    if i % 2 == 0:
        tbl3_lines.append(r"\rowcolor{rowlight}")
    tbl3_lines.append(row)
tbl3_body = "\n".join(tbl3_lines)

# ─────────────────────────────────────────────
# 7. SEKSI PER VARIABEL
# ─────────────────────────────────────────────
var_sections_parts = []
for var in vars_list:
    s = stats[var]
    label = var_label_tex[var]
    interp = desc_interp[var]
    unit = unit_labels[var]
    part_lines = [
        "",
        r"\subsection{" + label + "}",
        r"\begin{center}",
        r"\begin{tabular}{|l|r||l|r|}",
        r"\hline",
        r"\textbf{Statistik} & \textbf{Nilai} & \textbf{Statistik} & \textbf{Nilai} \\",
        r"\hline",
        f"$n$ (Observasi) & {int(s['n'])} & Rentang (Range) & {fmt(s['range'])} {unit} \\\\",
        f"Minimum & {fmt(s['min'])} {unit} & IQR ($Q_3 - Q_1$) & {fmt(s['iqr'])} {unit} \\\\",
        r"$Q_1$ (Kuartil 1) & " + f"{fmt(s['Q1'])} {unit} & Standar Deviasi & {fmt(s['std'])} {unit} \\\\",
        r"Median ($Q_2$) & " + f"{fmt(s['median'])} {unit} & Varian & {fmt(s['var'],2)} \\\\",
        r"Rata-rata ($\bar{x}$) & " + f"{fmt(s['mean'])} {unit} & Koef.~Variasi & {fmt(s['cv'],2)}\\% \\\\",
        r"$Q_3$ (Kuartil 3) & " + f"{fmt(s['Q3'])} {unit} & Skewness & {fmt(s['skew'])} \\\\",
        f"Maksimum & {fmt(s['max'])} {unit} & Kurtosis (Excess) & {fmt(s['kurt'])} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"\noindent " + interp,
        "",
    ]
    var_sections_parts.append("\n".join(part_lines))
var_sections_tex = "\n".join(var_sections_parts)

# ─────────────────────────────────────────────
# 8. MATRIKS KORELASI
# ─────────────────────────────────────────────
corr_rows = []
for i, v1 in enumerate(vars_list):
    row_vals = []
    for v2 in vars_list:
        r = pearson(raw[v1], raw[v2])
        if v1 == v2:
            row_vals.append(r"\textit{1.0000}")
        elif abs(r) > 0.5:
            row_vals.append(r"\textbf{" + f"{r:.4f}" + "}")
        else:
            row_vals.append(f"{r:.4f}")
    color = r"\rowcolor{rowlight}" + "\n" if i % 2 == 0 else ""
    corr_rows.append(color + var_nicknames[v1] + " & " + " & ".join(row_vals) + r" \\")
corr_body = "\n".join(corr_rows)

# Tabel pasangan signifikan
corr_interp_rows = []
interp_idx = 0
for i, v1 in enumerate(vars_list):
    for v2 in vars_list[i+1:]:
        r = pearson(raw[v1], raw[v2])
        if abs(r) >= 0.3:
            arah = "positif" if r > 0 else "negatif"
            kuat = "kuat" if abs(r)>=0.7 else ("sedang" if abs(r)>=0.5 else "lemah-sedang")
            color = r"\rowcolor{rowlight}" + "\n" if interp_idx % 2 == 0 else ""
            corr_interp_rows.append(
                color + f"{var_nicknames[v1]}--{var_nicknames[v2]} & {r:.4f} & Korelasi {arah} {kuat} \\\\"
            )
            interp_idx += 1
corr_interp_body = "\n".join(corr_interp_rows)

# ─────────────────────────────────────────────
# 9. TABLES FOR OUTLIERS & VIF
# ─────────────────────────────────────────────

# Outlier Table
tbl_outlier_lines = []
for i, var in enumerate(vars_list):
    s = stats[var]
    color = r"\rowcolor{rowlight}" + "\n" if i % 2 == 0 else ""
    tbl_outlier_lines.append(f"{color}{var_nicknames[var]} & {s['outlier_count']} & {s['outlier_pct']:.2f}\\% \\\\")
tbl_outlier_body = "\n".join(tbl_outlier_lines)

# VIF Table
tbl_vif_lines = []
for i, row in vif_df.iterrows():
    v = row['VIF']
    if v > 10:
        status = r"\textbf{Tinggi}"
    elif v > 5:
        status = r"\textbf{Sedang}"
    else:
        status = "Aman"
    
    color = r"\rowcolor{rowlight}" + "\n" if i % 2 == 0 else ""
    tbl_vif_lines.append(f"{color}{var_nicknames[row['Variable']]} & {v:.4f} & {status} \\\\")
tbl_vif_body = "\n".join(tbl_vif_lines)

SECTION_VISUALISASI = r"""
\newpage
%============================================================
\section{Visualisasi Distribusi dan Hubungan Spasial}
%============================================================

Visualisasi di bawah ini memberikan gambaran grafis mengenai persebaran data (Boxplot) dan hubungan antar variabel secara simultan (Pairplot).

\subsection{Boxplot Distribusi Variabel}
Boxplot digunakan untuk mengidentifikasi rentang data, kuartil, dan keberadaan pencilan (\textit{outliers}) secara visual. 

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{boxplot_deskriptif.png}
    \caption{Boxplot Seluruh Variabel Penelitian (Skala Mandiri)}
    \label{fig:boxplot_all}
\end{figure}

\subsection{Pairplot (Matrix Scatter Plot)}
Pairplot menyajikan matriks plot sebar untuk melihat hubungan linear antar pasangan variabel serta estimasi densitas (\textit{KDE}) pada diagonalnya.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{pairplot_deskriptif.png}
    \caption{Pairplot Matriks Hubungan Antar Variabel}
    \label{fig:pairplot_all}
\end{figure}
"""

SECTION_OUTLIER = r"""
\newpage
%============================================================
\section{Analisis Pencilan (Outlier Analysis)}
%============================================================

Identifikasi pencilan dilakukan menggunakan metode \textit{Interquartile Range} (IQR) dengan batas $Q_1 - 1,5 \times IQR$ dan $Q_3 + 1,5 \times IQR$.

\begin{table}[H]
\centering
\caption{Identifikasi Jumlah dan Persentase Outlier per Variabel}
\label{tab:outliers}
\begin{tabular}{lrr}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Variabel}} & \textcolor{white}{\textbf{Jumlah Outlier}} & \textcolor{white}{\textbf{Persentase}} \\
\midrule
""" + tbl_outlier_body + r"""
\bottomrule
\end{tabular}
\end{table}

\noindent \textbf{Catatan Penanganan Outlier:} 
Variabel harga ($Y$) dan luas ($X_1, X_2$) memiliki persentase pencilan yang signifikan. Hal ini mencerminkan karakteristik pasar properti Jakarta di mana terdapat hunian sangat mewah yang jauh melampaui rata-rata. Dalam analisis GWR, pencilan ini seringkali menjadi poin ketertarikan (\textit{local interest}) untuk melihat variasi spasial yang ekstrem.
"""

SECTION_VIF = r"""
\newpage
%============================================================
\section{Multikolinearitas dan VIF}
%============================================================

\subsection{Variance Inflation Factor (VIF)}
VIF mengukur seberapa besar varians dari koefisien regresi yang meningkat akibat adanya multikolinearitas.

\begin{table}[H]
\centering
\caption{Hasil Perhitungan Variance Inflation Factor (VIF)}
\label{tab:vif}
\begin{tabular}{lrl}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Variabel}} & \textcolor{white}{\textbf{Nilai VIF}} & \textcolor{white}{\textbf{Status}} \\
\midrule
""" + tbl_vif_body + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{Panduan Penanganan Korelasi Tinggi}
Berdasarkan nilai VIF dan matriks korelasi, jika ditemukan nilai VIF $> 10$ atau korelasi antar variabel independen $> 0{,}80$, beberapa langkah evaluasi yang dapat diambil adalah:

\begin{enumerate}
    \item \textbf{Eliminasi Variabel:} Menghapus salah satu variabel yang redundan (misal: antara Luas Bangunan dan Luas Tanah jika keduanya sangat korelasi).
    \item \textbf{Penggabungan (Rasio):} Mengubah variabel menjadi bentuk rasio, seperti Luas Bangunan per Jumlah Kamar.
    \item \textbf{Penelitian Spasial (GWR):} Pada model GWR, multikolinearitas lokal dapat terjadi secara dinamis. Jika multikolinearitas hanya terjadi di beberapa lokasi, variabel dapat tetap dipertahankan dengan catatan interpretasi pada wilayah tersebut dilakukan dengan hati-hati.
\end{enumerate}
"""

# ─────────────────────────────────────────────
# 10. SUSUN DOKUMEN LaTeX
# ─────────────────────────────────────────────
# Bangun sections secara terpisah untuk menghindari masalah escape dalam f-string

PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[indonesian]{babel}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{float}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{parskip}

\geometry{left=3cm, right=2.5cm, top=3cm, bottom=2.5cm}

\definecolor{headerblue}{RGB}{23, 62, 125}
\definecolor{rowlight}{RGB}{235, 242, 255}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\textcolor{headerblue}{\textbf{Statistika Deskriptif Detail}}}
\fancyhead[R]{\footnotesize\textcolor{gray}{Valuasi Harga Rumah DKI Jakarta 2024}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.5pt}

\titleformat{\section}{\Large\bfseries\color{headerblue}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\bfseries\color{headerblue!80!black}}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{headerblue!60!black}}{\thesubsubsection.}{0.5em}{}

\captionsetup{font=small,labelfont=bf}

\hypersetup{
  colorlinks=true,
  linkcolor=headerblue,
  urlcolor=blue,
  pdfauthor={Ammar Hanafi, Norman Mowlana Aziz, Kirono Dwi Saputro},
  pdftitle={Statistika Deskriptif Detail: Valuasi Harga Rumah DKI Jakarta 2024},
}

\begin{document}

\begin{titlepage}
\centering
\vspace*{1.5cm}
{\color{headerblue}\rule{\linewidth}{3pt}}\\[0.4cm]
{\LARGE\bfseries\color{headerblue} STATISTIKA DESKRIPTIF DETAIL}\\[0.3cm]
{\Large\bfseries Eksplorasi Karakteristik Data Uji}\\[0.15cm]
{\color{headerblue}\rule{\linewidth}{1pt}}\\[0.5cm]
{\large\textit{Analisis Geographically Weighted Regression (GWR)}\\
\textit{pada Kasus Valuasi Harga Rumah di DKI Jakarta 2024}}\\[2cm]

\begin{tabular}{c}
{\normalsize\textbf{AMMAR HANAFI (2206051582)}} \\[3pt]
{\normalsize\textbf{NORMAN MOWLANA AZIZ (2206025470)}} \\[3pt]
{\normalsize\textbf{KIRONO DWI SAPUTRO (2106656365)}} \\
\end{tabular}

\vfill
{\normalsize\textbf{PROGRAM STUDI SARJANA STATISTIKA}\\
\textbf{FAKULTAS MATEMATIKA DAN ILMU PENGETAHUAN ALAM}\\
\textbf{UNIVERSITAS INDONESIA}\\[0.3cm]
\textbf{DEPOK --- JUNI 2026}}\\[0.5cm]
{\color{headerblue}\rule{\linewidth}{2pt}}
\end{titlepage}

\tableofcontents
\newpage

%============================================================
\section{Pendahuluan dan Deskripsi Dataset}
%============================================================

Dokumen ini merupakan lampiran analitik khusus yang menyajikan eksplorasi statistik
deskriptif secara menyeluruh atas dataset harga rumah di DKI Jakarta tahun 2024.
Data bersumber dari \textit{GeoJSON dataset} pasar sekunder perumahan DKI Jakarta
yang telah melalui tahap \textit{data cleaning}. Total observasi yang valid adalah
\textbf{1.079 unit rumah}.

\subsection{Deskripsi Variabel Penelitian}

\begin{table}[H]
\centering
\caption{Definisi Operasional Variabel Penelitian}
\label{tab:var_def}
\small
\begin{tabular}{clll}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Notasi}} & \textcolor{white}{\textbf{Nama Variabel}} & \textcolor{white}{\textbf{Satuan}} & \textcolor{white}{\textbf{Peran}} \\
\midrule
\rowcolor{rowlight}
$Y$   & Harga Rumah         & Miliar Rp & Dependen                   \\
$X_1$ & Luas Bangunan       & m$^2$     & Independen (Fisik)         \\
\rowcolor{rowlight}
$X_2$ & Luas Tanah/Kaveling & m$^2$     & Independen (Fisik)         \\
$X_3$ & Jumlah Kamar        & Unit      & Independen (Fisik)         \\
\rowcolor{rowlight}
$X_4$ & Jarak ke Monas      & km        & Independen (Spasial)       \\
$X_5$ & Jarak ke Sekolah    & km        & Independen (Aksesibilitas) \\
\rowcolor{rowlight}
$X_6$ & Jarak ke Rumah Sakit & km       & Independen (Aksesibilitas) \\
$X_7$ & Jarak ke Jalan Utama & km       & Independen (Aksesibilitas) \\
\bottomrule
\end{tabular}
\end{table}

\newpage
%============================================================
\section{Statistik Ringkasan Komprehensif}
%============================================================

\subsection{Ukuran Pemusatan dan Posisi}

\begin{table}[H]
\centering
\caption{Lima Angka Ringkasan, Rata-rata, dan Standar Deviasi}
\label{tab:summary1}
\small
\setlength{\tabcolsep}{5.5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Var}} &
\textcolor{white}{\textbf{$n$}} &
\textcolor{white}{\textbf{Min}} &
\textcolor{white}{\textbf{$Q_1$}} &
\textcolor{white}{\textbf{Median}} &
\textcolor{white}{\textbf{Mean}} &
\textcolor{white}{\textbf{$Q_3$}} &
\textcolor{white}{\textbf{Max}} &
\textcolor{white}{\textbf{Std.Dev}} \\
\midrule
"""

PREAMBLE_END_TBL1 = r"""
\bottomrule
\end{tabular}
\end{table}

\noindent\textbf{Keterangan:} $Q_1$ = Kuartil ke-1 (persentil ke-25),
$Q_3$ = Kuartil ke-3 (persentil ke-75).
Nilai mean $Y$ yang jauh di atas median mengkonfirmasi distribusi harga yang menceng
kanan (\textit{positively skewed}) --- ciri khas pasar properti heterogen.

\subsection{Ukuran Keragaman dan Dispersi}

\begin{table}[H]
\centering
\caption{Ukuran Dispersi dan Keragaman Data}
\label{tab:dispersi}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrrrr}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Var}} &
\textcolor{white}{\textbf{Varian ($s^2$)}} &
\textcolor{white}{\textbf{CV (\%)}} &
\textcolor{white}{\textbf{IQR}} &
\textcolor{white}{\textbf{Range}} &
\textcolor{white}{\textbf{Skewness}} &
\textcolor{white}{\textbf{Kurtosis*}} \\
\midrule
"""

PREAMBLE_END_TBL2 = r"""
\bottomrule
\multicolumn{7}{l}{\footnotesize *Excess Kurtosis (Normal$=0$).
  Positif $\Rightarrow$ leptokurtik (ekor tebal), Negatif $\Rightarrow$ platikurtik.}
\end{tabular}
\end{table}

\noindent\textbf{Interpretasi CV:} Koefisien Variasi (CV) mengukur keragaman relatif terhadap
rata-rata. CV $>30\%$ = keragaman \textit{tinggi}. Seluruh variabel memiliki CV relatif
tinggi, mencerminkan heterogenitas data perumahan Jakarta yang nyata.

\subsection{Klasifikasi Bentuk Distribusi}

\begin{table}[H]
\centering
\caption{Klasifikasi Distribusi Berdasarkan Momen Ketiga dan Keempat}
\label{tab:distribusi}
\small
\begin{tabular}{lrrlr}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Var}} &
\textcolor{white}{\textbf{Skewness}} &
\textcolor{white}{\textbf{Kurtosis}} &
\textcolor{white}{\textbf{Kemencengan}} &
\textcolor{white}{\textbf{Ketinggian Puncak}} \\
\midrule
"""

SECTION3_HEADER = r"""
\bottomrule
\end{tabular}
\end{table}

\newpage
%============================================================
\section{Analisis Per Variabel}
%============================================================

Berikut disajikan analisis statistik deskriptif mendalam per variabel, meliputi
tabel statistik lengkap dan interpretasi analitik.

"""

SECTION4_HEADER = r"""
\newpage
%============================================================
\section{Analisis Korelasi Antar Variabel}
%============================================================

\subsection{Matriks Korelasi Pearson}

\begin{table}[H]
\centering
\caption{Matriks Korelasi Pearson Antar Variabel (dihitung dari data mentah)}
\label{tab:corr_full}
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{}} &
\textcolor{white}{\textbf{Y}} &
\textcolor{white}{\textbf{X1}} &
\textcolor{white}{\textbf{X2}} &
\textcolor{white}{\textbf{X3}} &
\textcolor{white}{\textbf{X4}} &
\textcolor{white}{\textbf{X5}} &
\textcolor{white}{\textbf{X6}} &
\textcolor{white}{\textbf{X7}} \\
\midrule
"""

SECTION4_MID = r"""
\bottomrule
\multicolumn{9}{l}{\footnotesize \textbf{Tebal} = korelasi kuat ($|r|>0{,}5$)}
\end{tabular}
\end{table}

\subsection{Pasangan Korelasi Bermakna}

\begin{table}[H]
\centering
\caption{Pasangan Variabel dengan Korelasi Bermakna ($|r|\geq0{,}30$)}
\label{tab:corr_interp}
\small
\begin{tabular}{lrl}
\toprule
\rowcolor{headerblue}
\textcolor{white}{\textbf{Pasangan}} &
\textcolor{white}{\textbf{$r$ Pearson}} &
\textcolor{white}{\textbf{Interpretasi}} \\
\midrule
"""

CLOSE_TABLE = r"""
\bottomrule
\end{tabular}
\end{table}
"""

SECTION4_END = r"""
\subsection{Temuan Utama Korelasi}

\begin{enumerate}
  \item \textbf{Fisik $\to$ Harga (Positif Kuat):}
    Luas Bangunan ($X_1$) memiliki korelasi Pearson tertinggi dengan harga
    ($r\approx0{,}54$), diikuti Luas Tanah ($X_2$, $r\approx0{,}52$) dan
    Jumlah Kamar ($X_3$, $r\approx0{,}34$). Konsisten dengan teori valuasi
    hedonis bahwa dimensi fisik menjadi faktor penentu utama.

  \item \textbf{Multikolinearitas Fisik:}
    Korelasi sangat kuat antara $X_1$--$X_2$ ($r\approx0{,}80$) dan
    $X_1$--$X_3$ ($r\approx0{,}60$) mengindikasikan potensi
    \textit{multikolinearitas} dalam model OLS global.

  \item \textbf{Fasilitas Publik Terklasterisasi:}
    Korelasi $X_5$--$X_6$ ($r\approx0{,}75$) sangat tinggi --- sekolah dan
    rumah sakit cenderung berlokasi berdekatan, mengkonfirmasi pola zonasi
    fasilitas publik di Jakarta.

  \item \textbf{Jarak Negatif ke Harga:}
    $X_4$ dan $X_6$ berkorelasi negatif dengan $Y$, meski lemah secara global.
    Heterogenitas spasialnya dikaptulasi lebih baik oleh model GWR lokal.

  \item \textbf{Jalan Utama Hampir Tidak Berkorelasi:}
    $X_7$ memiliki korelasi mendekati nol dengan $Y$ ($r\approx0{,}009$),
    kemungkinan karena hampir semua properti memiliki akses jalan yang setara.
\end{enumerate}

\newpage
%============================================================
\section{Catatan Metodologis}
%============================================================

\subsection{Rumus yang Digunakan}

Seluruh ukuran statistik deskriptif dihitung menggunakan formula standar berikut:

\begin{align*}
\text{Rata-rata:} \quad & \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i \\[6pt]
\text{Varian (tidak bias):} \quad & s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2 \\[6pt]
\text{Koef. Variasi:} \quad & \text{CV} = \frac{s}{|\bar{x}|} \times 100\% \\[6pt]
\text{Skewness (Fisher):} \quad & g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3 \\[6pt]
\text{Excess Kurtosis:} \quad & g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n}\left(\frac{x_i-\bar{x}}{s}\right)^4 \\
& \quad - \frac{3(n-1)^2}{(n-2)(n-3)} \\[6pt]
\text{Korelasi Pearson:} \quad & r_{XY} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
  {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\;\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}} \\[6pt]
\text{IQR:} \quad & \text{IQR} = Q_3 - Q_1 \\[6pt]
\text{Persentil (Interp.\ Linear):} \quad & P_p = x_{(L)} + \text{frac}\cdot(x_{(L+1)}-x_{(L)})
\end{align*}

\subsection{Catatan Implementasi}
\begin{itemize}
  \item Varian dihitung dengan \textit{estimator tidak bias} (pembagi $n-1$).
  \item Skewness dan kurtosis menggunakan koreksi Fisher--Pearson (\textit{adjusted}).
  \item Persentil menggunakan metode interpolasi linear.
  \item Data diekstrak langsung dari \texttt{rumah\_27\_8\_25\_clean.geojson} tanpa transformasi.
\end{itemize}

\newpage
%============================================================
\section{Lampiran: Akses Data Penelitian}
%============================================================

Seluruh dataset primer yang digunakan dalam analisis ini, mencakup file GeoJSON mentah,
dataset Excel hasil pembersihan, serta script pemrosesan data, dapat diakses
melalui repositori daring berikut:

\begin{center}
\textbf{Link Dataset (Google Drive):}\\
{\raggedright \url{https://drive.google.com/drive/folders/1dmIp66Qh8gmt7BZdckrE9OWTKQKyhDJV?usp=sharing} \par}
\end{center}

Dataset ini disediakan untuk kepentingan replikasi penelitian dan transparansi
metodologi pemodelan spasial.

\end{document}
"""

# ─────────────────────────────────────────────
# 10. GABUNGKAN SEMUA
# ─────────────────────────────────────────────
latex_content = (
    PREAMBLE +
    tbl1_body +
    PREAMBLE_END_TBL1 +
    tbl2_body +
    PREAMBLE_END_TBL2 +
    tbl3_body +
    SECTION3_HEADER +
    var_sections_tex +
    SECTION_VISUALISASI +
    SECTION_OUTLIER +
    SECTION4_HEADER +
    corr_body +
    SECTION4_MID +
    corr_interp_body +
    CLOSE_TABLE +
    SECTION_VIF +
    SECTION4_END
)

# ─────────────────────────────────────────────
# 11. TULIS & COMPILE
# ─────────────────────────────────────────────
tex_file = "desc_stats_detail.tex"
with open(tex_file, "w", encoding="utf-8") as f:
    f.write(latex_content)

print(f"LaTeX ditulis: {tex_file}")
print("Mengkompilasi PDF (2 pass untuk TOC)...")

ok = True
for run in range(2):
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_file],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if result.returncode != 0 and run == 0:
        ok = False
        print("=== LOG (last 4000 chars) ===")
        print(result.stdout[-4000:])
        break

pdf = "desc_stats_detail.pdf"
if os.path.exists(pdf):
    size_kb = os.path.getsize(pdf) // 1024
    print(f"\n SUCCESS! {pdf} ({size_kb} KB)")
else:
    print("\n GAGAL. Cek log di atas.")
