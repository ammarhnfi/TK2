import pandas as pd
import sys

try:
    df = pd.read_excel('Ringkasan_Paper_1-6.xlsx')
except Exception as e:
    print('Gagal membaca Excel:', e)
    sys.exit(1)

langkah_col = [c for c in df.columns if 'langkah' in c.lower()]

if not langkah_col:
    print('Kolom langkah-langkah tidak ditemukan!')
    sys.exit(1)

langkah_baru = [
    # Paper 1
    "1. Pengumpulan data tuberkulosis dan variabel prediktor dari Dinkes Makassar.\n2. Pengujian multikolinieritas antar variabel.\n3. Pemodelan awal menggunakan Regresi Poisson Global.\n4. Pengujian efek heterogenitas spasial dengan tes Breusch-Pagan.\n5. Kalibrasi model GWPR menggunakan pembobot Fixed Gaussian dan Fixed Bi-square.\n6. Pemilihan model terbaik berdasarkan kriteria AIC dan R2.",

    # Paper 2
    "1. Tinjauan model ekonometrik spasial mainstream (SAR, SEM, SDM) dan kelemahannya.\n2. Pengenalan model SLX (Spatial Lag of X) sebagai standar dasar keterkaitan spasial.\n3. Aplikasi empiris model SLX pada data riil panel permintaan rokok di negara bagian AS.\n4. Komparasi performa fungsi parameter yang memuat efek spillover antar wilayah.\n5. Demonstrasi pembuktian nilai manfaat matriks pembobot spasial yang difleksibelkan.",

    # Paper 3
    "1. Identifikasi kebutuhan permodelan hierarki ganda (multilevel) pada korelasi spasial tanah mikro dan makro di Beijing.\n2. Formulasi persamaan teoritik regresi keruangan Hierarchical Spatial Autoregressive (HSAR).\n3. Desain komputasi optimasi estimator menggunakan iterasi Bayesian Markov Chain Monte Carlo (MCMC).\n4. Isolasi signifikansi variasi bauran keruangan spesifik level persil regional dan area distrik kota.\n5. Evaluasi stabilitas minimalisasi koefisien bias komputasional MCMC dalam penaksiran proporsional.",

    # Paper 4
    "1. Diagnosis objektif kelemahan model regresi Random Effects Eigenvector Spatial Filtering (RE-ESF) terhadap spatial confounders.\n2. Modifikasi konstruksi variabel eigenvector dari matriks konektivitas spasial C yang dihaluskan.\n3. Formulasi fungsi objektif optimasi model (Residual Maximum Likelihood/REML).\n4. Pembuatan skenario pengujian sintetik menggunakan simulasi Data Generating Process (DGP) Monte Carlo terkontrol.\n5. Kalkulasi penurunan galat parameter (RMSE) untuk memvalidasi efisiensi dari usulan model RE-ESF ekstensi baru.",

    # Paper 5
    "1. Pengumpulan empiris data atribut hedonik lokasi observasi harga rumah di London pada tahun 2001.\n2. Konversi kedekatan batas jarak Euclidean konvensional ke dalam matriks aspal nyata (Network Distance) dan Waktu Tempuh aspal (Travel Time).\n3. Formulasi kriteria optimalisasi pita bandwidth secara adaptif dan absolut pada fungsi model lokal.\n4. Kalibrasi Geographically Weighted Regression (GWR) ke semua simulasi pengujian skenario jarak yang difilter kriteria AICc.\n5. Pemilihan akurasi kemampuan model prediktif harga properti dengan R2 paling mendekati realita.",

    # Paper 6
    "1. Penguraian rasional teoretis atas kesalahan interpretasi koefisien variabel spesifikasi tunggal analisis keruangan layaknya OLS biasa.\n2. Rasionalisasi matematis fungsi kalkulus marginal dampak spasial (Direct, Indirect, dan Spillover Effects) berdasarkan inverse pengali spasial.\n3. Pembuatan rekayasa stokastik parameter acak DGP Monte Carlo dengan meminjam lekukan kontur geometri 80 map Michigan empiris sesungguhnya.\n4. Pengujian 5 spektrum multivariat autokorelasi simultan (SAR, SEM, SDM, SDEM, SAC) menggunakan dummy spesimen matriks buatan.\n5. Pembacaan akhir derivasi turunan level signifikansi tiap rata-rata efek parsial LeSage."
]

col_name = langkah_col[0]
df[col_name] = langkah_baru

try:
    df.to_excel('Ringkasan_Paper_1-6.xlsx', index=False)
    print('Berhasil memperbarui file Excel!')
except Exception as e:
    print('Gagal menyimpan file:', e)
