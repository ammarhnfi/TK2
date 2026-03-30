import pandas as pd

file_path = 'd:/tk2/3_Paper_Summary/Ringkasan_Paper_7-12.xlsx'
df = pd.read_excel(file_path)

# 7
idx = 0
df.loc[idx, 'Pengarang & Tahun'] = 'Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996)'
df.loc[idx, 'Judul Paper'] = 'Geographically weighted regression: a method for exploring spatial nonstationarity'
df.loc[idx, 'Tujuan Penelitian'] = 'Memperkenalkan dan mengembangkan teknik Geographically Weighted Regression (GWR) sebagai solusi empiris untuk menganalisis variasi hubungan spasial (nonstasioneritas), di mana variabel independen dapat berdampak berbeda di setiap titik lokasi kajian, mengatasi batasan model global yang mengasumsikan parameter konstan di seluruh ruang.'
df.loc[idx, 'Data dan Variabel'] = 'Data: Data sensus perumahan riil di kawasan Tyne and Wear, Inggris tahun 1991.\nVariabel: Harga jual perumahan (dependent), tipe kepemilikan, pendapatan, dan jumlah kendaraan bermotor.'
df.loc[idx, 'Metode'] = 'Geographically Weighted Regression (GWR) dengan Fungsi Pembobot Kernel.'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mengkritisi limitasi dari OLS dan model ekspansi teritorial (expansion method).\n2. Memformulasikan estimasi GWR berdasarkan titik geografis sampel.\n3. Merancang fungsi pembobotan spasial (kernel regression) dimana bobot menurun seiring jarak.\n4. Menerapkan uji statistik Monte Carlo untuk menguji signifikansi nonstasioneritas dari koefisien regresi.\n5. Mengaplikasilkan metode tersebut pada data harga rumah empirik dan memetakan koefisien GWR.'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: Ditemukan bahwa efek pendapatan terhadap probabilitas kepemilikan mobil berbanding lurus, namun magnitudo efeknya sangat bervariasi di berbagai bagian wilayah (spatially non-stationary).\nKesimpulan: GWR adalah metode eksploratif spasial yang jauh lebih kuat dibandinkan regresi global, mampu mengungkap pola heterogenitas spasial tersembunyi yang krusial untuk perumusan kebijakan lokal.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Mengeksplorasi penggunaan matriks jarak jaringan jalan (network distance) alih-alih jarak Euclidean dalam pembobotan kernel GWR untuk memodelkan fenomena urban yang lebih akurat.'

# 8
idx = 1
df.loc[idx, 'Pengarang & Tahun'] = 'Lee, L. F., & Yu, J. (2010)'
df.loc[idx, 'Judul Paper'] = 'Estimation of spatial autoregressive panel data models with fixed effects'
df.loc[idx, 'Tujuan Penelitian'] = 'Mengembangkan dan membuktikan metode estimasi Maximum Likelihood (MLE) yang konsisten untuk Model Data Panel Autoregresif Spasial (SAR) yang mengandung "fixed effects" individu maupun waktu, sekaligus mengatasi masalah inkonsistensi (incidental parameter problem).'
df.loc[idx, 'Data dan Variabel'] = 'Data: Eksperimen simulasi Monte Carlo untuk memvalidasi pembuktian asimtotik matematis dengan berbagai dimensi T (waktu) dan N (individu).\nVariabel: Koefisien autoregresif spasial (rho), koefisien regresi variabel independen (beta), dan parameter varians error (sigma squared).'
df.loc[idx, 'Metode'] = 'Quasi-Maximum Likelihood Estimation (QMLE) dengan Transformasi Orthonormal bias-corrected.'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mendemonstrasikan bahwa estimasi MLE langsung pada model panel spasial efek tetap menderita inkonsistensi jika T kecil (incidental parameter problem).\n2. Mengajukan pendekatan transformasi data untuk mengeliminasi fixed effects sebelum estimasi MLE dilakukan.\n3. Memformulasikan kondisi keterbatasan asimtotik dari distribusi estimator.\n4. Merumuskan koreksi bias analitik (bias-corrected estimator) ketika model mengandung baik individual maupun time effects.\n5. Mensimulasikan data untuk memverifikasi akurasi estimator yang diusulkan.'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: Transformasi orthonormal sukses mengeliminasi fixed effect tanpa memicu bias pada parameter utama. Koreksi bias yang diformulasikan efektif memulihkan parameter varians yang sebelumnya under-estimated saat periode waktu panel pendek.\nKesimpulan: Untuk menghindari inferensi spasial yang keliru akibat parameter incidental, QMLE pada data panel spasial harus dilakukan melalui transformasi efek-tetap yang diusulkan, disertai koreksi bias asimtotik.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Mengaplikasikan QMLE bias-corrected pada model Durbin Panel Spasial (SDM) yang memuat limpahan heterogen pada evaluasi dampak ekonomi lintas provinsi di negara berkembang.'

# 9
idx = 2
df.loc[idx, 'Pengarang & Tahun'] = 'Kelejian, H. H., & Piras, G. (2014)'
df.loc[idx, 'Judul Paper'] = 'Estimation of spatial models with endogenous weighting matrices, and an application to a demand model for cigarettes'
df.loc[idx, 'Tujuan Penelitian'] = 'Menyelesaikan tantangan estimasi dalam regresi spasial dimana matriks pembobot ketetanggaan (W) tidak bebas (endogen) vis-a-vis gangguan error model, dengan menyodorkan prosedur kerangka variabel instrumen (GS2SLS).'
df.loc[idx, 'Data dan Variabel'] = 'Data: Ilustrasi empiris menggunakan panel data permintaan rokok lintas negara bagian di AS (1963–1992).\nVariabel: Penjualan rokok per kapita, harga rokok, rasio populasi, pendapatan. Matriks W didasarkan dari bobot endogen jarak jarak sosio-ekonomis perbatasan.'
df.loc[idx, 'Metode'] = 'Generalized Spatial Two-Stage Least Squares (GS2SLS).'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mendefinisikan model ekonometrika spasial di mana elemen matriks W dikonstruksi secara stokastik (mis. volume perdagangan bilateral/arus migrasi) yang rentan berkorelasi dengan regresi error.\n2. Mendeduksi ekspansi Taylor series element-by-element atas matriks W guna melerai komponen instrumen basis struktural pembobot endogen.\n3. Mengajukan estimator Generalized Spatial Two-Stage Least Squares (GS2SLS) berbasis GMM.\n4. Menguji asimtotik normalitas dan konsistensi matriks varians-kovarians estimator.\n5. Menerapkan pengujian teoretis pada isu pergeseran pembelian rokok lintas-batas federal (bootlegging).'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: Pengabaian endogenitas pada formasi matriks W (seperti jarak sosio-ekonomi atau konektivitas perdagangan) menyebabkan estimator MLE klasik berbias hebat. Algoritma GS2SLS dapat menangani hal tersebut secara andal (robust) dan menghasilkan estimasi parameter yang mendekati nilai sejati DGP-nya.\nKesimpulan: Penggunaan variabel pembobot yang mengandung perilaku simutan harus distrukturkan ulang lewat pendekatan instrumen spasial yang diusulkan, bukan MLE standar.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Merancang instrumen GS2SLS untuk menyelesaikan permasalahan matriks jarak W berbasis sentimen di Twitter dalam memodelkan penyebaran efek bola salju opini politik antar-daerah secara spasial.'

# 10
idx = 3
df.loc[idx, 'Pengarang & Tahun'] = 'Oshan, T. M., Li, Z., Kang, W., Wolf, L. J., & Fotheringham, A. S. (2019)'
df.loc[idx, 'Judul Paper'] = 'mgwr: A Python implementation of multiscale geographically weighted regression for investigating process spatial heterogeneity and scale'
df.loc[idx, 'Tujuan Penelitian'] = 'Menawarkan "blue-print" pertama dan merilis paket komputasi efisien berbasis Python untuk model Multiscale Geographically Weighted Regression (MGWR), agar parameter regresi dapat bervariasi pada tingkatan bandwidth spatial yang saling terisolasi dan spesifik bagi setiap prediktor individual.'
df.loc[idx, 'Data dan Variabel'] = 'Data: Data simulasi stasioner, data terapan Georgia (pendidikan) dan Dublin (pemilihan umum lokal).\nVariabel: Proporsi kepemilikan gelar tinggi, usia, ras, persentase disabilitas dalam dataset Georgia.'
df.loc[idx, 'Metode'] = 'Multiscale Geographically Weighted Regression (MGWR) komputasional via Algoritma Iteratif Backfitting.'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mengulas masalah pada GWR klasik yang mendesak sebuah bandwidth skala tunggal secara global (over/under-smoothing parameter).\n2. Membangun dan mengimplementasikan algoritma interatif konvergen Backfitting untuk ekstraksi bandwidth spesifik per variabel bebas.\n3. Mengkodifikasi diagnostik MGWR baru untuk inferensi p-value (mengoreksi t-values akibat dependence) via False Discovery Rate (FDR).\n4. Menyajikan arsitektur paket perangkat lunak Python (mgwr) dan mencontohkan optimalisasi multi-processing untuk meminimalkan durasi eksekusi algoritma kernel search.\n5. Menjalankan studi kasus demografis sebagai bukti keunggulan empiris (empirical benchmarking).'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: MGWR secara stabil menurunkan bias regresi dan mencegah munculnya kolinearitas artifisial ("concurvity") ketimbang GWR tradisional. Skala relasi demografis Georgia terbukti tidak seragam: usia berdampak stabil global, sementara proporsi ras minoritas memicu impak dengan resolusi sangat lokal.\nKesimpulan: MGWR adalah default baru pemodelan lokal; pustaka mgwr menjamin efisiensi asimtotik dan fleksibilitas bagi peneliti big data geospasial menelusuri fenomena multiskala secara independen.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Mengkolaborasikan pustaka MGWR dengan integrasi algoritma klasterisasi spatial machine-learning (seperti HDBSCAN) atas parameter regresi spasial multiskalanya untuk segmentasi pasar properti presisi tinggi.'

# 11
idx = 4
df.loc[idx, 'Pengarang & Tahun'] = 'Wheeler, D., & Tiefelsdorf, M. (2005)'
df.loc[idx, 'Judul Paper'] = 'Multicollinearity and correlation among local regression coefficients in geographically weighted regression'
df.loc[idx, 'Tujuan Penelitian'] = 'Menyelidiki malfungsi potensial yang terjadi secara intrinsik ketika model GWR memicu artifak ketergantungan (multikolinearitas eksesif) antar koefisien parameter spasial di sekitar satu lokasi geografis, yang berpotensi mencederai validitas kesimpulan variabilitas spasial.'
df.loc[idx, 'Data dan Variabel'] = 'Data: Pengujian Monte Carlo atas matriks desain prediktor spasial tiruan, serta kasus empiris distribusi angka kanker di negara bagian AS (Data demografi lansia & kemiskinan).\nVariabel: VIF lokal dari koefisien spasial (Condition Number), matriks pembobot kernel, dan koefisien regresi.'
df.loc[idx, 'Metode'] = 'Geographically Weighted Regression dengan uji korelasi artifisial vektor ortogonal (Local Collinearity Diagnostics).'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mendemonstrasikan bahwa koefisien GWR acapkali menampilkan hubungan korelasional sangat kuat yang menakutkan antar-variabel dependen.\n2. Merumuskan serangkaian DGP (Data Generating Process) di mana variabel eksogen di-setting ekuivalen ortogonal sempurna (independen total) untuk menyaring korelasi tulen dan menelanjangi korelasi buatan (spurious).\n3. Mengeksekusi regresi GWR pada set data simulatif dengan berbagai seleksi resolusi bandwidth kernel dan bentuk matriks geometris lattice.\n4. Menghitung korelasi Pearson antar profil koefisien spasial, lalu memvalidasinya dengan indeks condition number.'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: Pembobotan data yang dioverlapkan pada kernel spasial GWR membuahkan koefisien regresi spasial yang saling berafiliasi negatif/positif drastis (Pearson >0.8), kendati asal datanya dirancang benar-benar tidak kolinear.\nKesimpulan: Peta variasi koefisien yang diproduksi GWR harus disikapi sangat skeptis atau dipandang berbahaya jika tidak melewati filter metrik diagnosis VIF spasial yang diajukan. Peneliti pantang menarik vonis substantif dari koefisien lokal GWR yang terjangkit kolinear lokal artifisial ini.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Menciptakan penalti regularisasi (Ridge/Lasso GWR) pada arsitektur fungsi lokal GWR spesifik untuk memutus mata rantai korelasi artifisial akibat pergerakan iteratif kalibrasi matriks bobot.'

# 12
idx = 5
df.loc[idx, 'Pengarang & Tahun'] = 'LeSage, J. P., & Pace, R. K. (2014)'
df.loc[idx, 'Judul Paper'] = 'The biggest myth in spatial econometrics'
df.loc[idx, 'Tujuan Penelitian'] = 'Membongkar miskonsepsi (mitos) lazim di kalangan ilmuwan empiris bahwa temuan penelitian regresi spasial luar biasa rapuh dan super-sensitif terhadap keputusan subyektif pemilihan matriks pembobot ketetanggaan spasial W (contoh W orde-1 vs. Inverse Distance).'
df.loc[idx, 'Data dan Variabel'] = 'Data: Simulasi Monte Carlo ekstensif menggunakan matriks perbatasan spasial negara bagian AS dan county.\nVariabel: Marginal Average Direct Effect (ADE), Average Indirect Effect (AIE - limpahan), dan Total Effect berbasis efek kalkulasi turunan parsial ruang DGP.'
df.loc[idx, 'Metode'] = 'Spatial Autoregressive Model (SAR) dan Spatial Durbin Model (SDM) beserta Penguraian Impak Turunan Parsial Iteratif (Marginal Direct/Indirect Summary).'
df.loc[idx, 'Langkah-langkah Penelitian'] = '1. Mengidentifikasi asal-usul mispersepsi bahwa peneliti yang membanca koefisien point-estimate (titik beta) akan selalu menggenerasi koefisien yang aneh bila konfigurasi matriks W diubah.\n2. Menstrukturkan persamaan kalkulasi total impact regresi spasial yang mengikutsertakan matriks leontief inverse (spillover iteratif).\n3. Me-running seribu simulasi SDM spasial secara kontras: menghasilkan data berbekal struktur jarak kuadrat (inverse distance) tapi mengestimasinya menggunakan sekadar matriks tetangga singgungan biner (contiguity).\n4. Membandingkan bias komputasional dari ringkasan parsial Average Direct/Indirect effect antara model salah sel (misspecified W) dengan spesifikasi yang betul.'
df.loc[idx, 'Hasil dan Kesimpulan'] = 'Hasil: Perbedaan matriks bobot W nyaris tidak mengubah besaran efek marjinal limpahan tak langsung maupun langsung secara material, selama spesifikasi regresi (variabel eksogen dsb.) mapan, lantaran parameter dispersi (rho) sanggup mengkompensasi variasi kepadatan tetangga di dalam matriks W.\nKesimpulan: Mitos sensitivitas mematikan W timbul dari metodologi evaluasi naif, yaitu penyamaan interpretasi "point estimates beta" ala OLS global terhadap formula spasial leontief lokal. Parameter total limpahan (total effect spatial summary) lah yang konkrit menjamin daya robus dari metode regresi spasial model Durbin maupun SAR.'
df.loc[idx, 'Ide Baru untuk Penelitian'] = 'Melakukan re-studi validasi paper ini untuk menelaah robusitas efek kalkulasi turunan parsial lintas matriks pada estimasi regresi spatial Bayesian di negara kepulauan berdensitas terfragmentasi asimetris ekstrim tinggi.'

df.to_excel('d:/tk2/3_Paper_Summary/Ringkasan_Paper_7-12_Lengkap.xlsx', index=False)
print("Updated comprehensions")
