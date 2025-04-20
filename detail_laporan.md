# Laporan Proyek Machine Learning - Rafif Idris Ardhana

## Domain Proyek: Kesehatan
Perusahaan asuransi kesehatan adalah lembaga yang menyediakan perlindungan finansial bagi individu terhadap biaya medis yang tidak terduga. Suatu perusahaan asuransi bertanggung jawab untuk menanggung sebagian atau seluruh biaya perawatan medis yang timbul akibat penyakit atau kecelakaan individu/kelompok. Bagi suatu perusahaan asuransi penting untuk memperkirakan kisaran biaya yang perlu dipersiapkan untuk individu/kelompok di masa depan. Perkiraan atau prediksi ini memungkinkan perusahaan asuransi menyesuaikan premi secara adil dan efisien.

**Rubrik/Kriteria Tambahan**:
- Masalah ini sangat penting bagi banyak pemangku kepentingan perusahaan kesehatan khususnya asuransi, sebab estimasi biaya yang akurat dapat membantu perusahaan untuk merencanakan masa depan dan memprioritaskan alokasi sumber daya manajemen. Masalah ini dapat diselesaikan dengan mengadopsi teknologi terkini yaitu Machine Learning. Machine Learning (ML) merupakan salah satu aspek kecerdasan komputasional yang dapat memecahkan berbagai masalah dalam berbagai aplikasi dan sistem dalam hal memanfaatkan data historis. Khususnya di sektor asuransi, ML dapat membantu meningkatkan efisiensi penyusunan kebijakan termasuk premi, klaim, dan lainnya. Algoritma ML sangat baik dalam memprediksi pengeluaran pasien yang kemungkinan berbiaya tinggi dan sangat membutuhkan.
- Referensi:
  ul Hassan, C. A., Iqbal, J., Hussain, S., AlSalman, H., Mosleh, M. A., & Sajid Ullah, S. (2021). A computational intelligence approach
      for predicting medical insurance cost. Mathematical Problems in Engineering, 2021
  Diakses melalui: https://onlinelibrary.wiley.com/doi/abs/10.1155/2021/1162553

## Business Understanding

Suatu perusahaan asuransi kesehatan memiliki tanggung jawab yang besar terhadap perjanjian medis yang dilakukan dengan individu/kelompok. Perusahaan akan mengalami kerugian signifikan jika salah menentukan premi asuransi. Misalnya premi yang terlalu rendah akan menyebabkan kerugian finansial saat klaim melonjak, sedangkan premi yang terlalu tinggi menyebabkan kehilangan pelanggan akibat nilai yang tidak kompetitif. Selain itu, perusahaan asuransi tersebut juga tidak dapat mengidentifikasi pasien dengan resiko tinggi yang berkemungkinan dapat ditawarkan program kesehatan khusus.

Oleh karena itu, penting bagi perusahaan asuransi kesehatan untuk mengetahui dan dapat memprediksi kejadian di masa depan dengan menggunakan suatu sistem. Hal ini akan berdampak positif pada penyusunan anggaran, perencanaan risiko, dan penyesuaian strategi akan mengandalkan teknologi atau automation judgement, yang mana dapat membantu pengambilan keputusan dan prosesnya tidak memakan waktu lama.

### Problem Statements
- Dari serangkaian atribut(fitur) yang ada, atribut(fitur) apa yang paling berpengaruh terhadap biaya medis (charge) asuransi?
- Bagaimana cara memprediksi biaya medis (charges) yang akan dikeluarkan seseorang berdasarkan fitur yang ada?

### Goals
- Mengetahui atribut(fitur) yang paling berkorelasi dengan biaya medis (charges)
- Membuat machine learning yang dapat memprediksi biaya medis (charges) berdasarkan fitur yang ada

**Rubrik/Kriteria Tambahan**:
### Solution statements
- Membangun model baseline menggunakan Linear Regression.
- Membangun model alternatif menggunakan Random Forest Regressor dan Gradient Boosting Regressor.
- Mengukur performa model menggunakan MAE, MSE, dan R2 Score.
- Memilih model terbaik berdasarkan nilai MAE dan MSE terendah serta R2 Score tertinggi

## Data Understanding
Dataset yang digunakan merupakan dataset publik yang banyak digunakan dalam pembelajaran machine learning, terutama untuk studi kasus regresi yaitu Medical Cost Personal Dataset yang diperoleh dari Kaggle.
- Dataset memiliki jumlah sampel 1.338 baris.
- Terdapat beberapa fitur non-numerik seperti "sex", "smoker", "region".
- Beberapa fitur numerik seperti "age", "bmi", "children", "charges".
- Berdasarkan struktur penamaan wilayah (southeast, southwest, northeast, northwest) serta tingginya nilai biaya medis (charges), dataset ini sangat besar kemungkinan berasal dari konteks layanan kesehatan di Amerika Serikat.

### Variabel-variabel pada Medical Cost Personal dataset adalah sebagai berikut:
- Age: usia pelanggan/pasien
- Sex: jenis kelamin pelanggan/pasien
- Bmi: indeks masa tubuh pelanggan/pasien
- Children: tanggungan anak pelanggan/pasien
- Smoker: status perokok pelanggan/pasien
- Region: wilayah tempat tinggal pelanggan/pasien
- Charges: biaya medis pelanggan/pasien

Link dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download

**Rubrik/Kriteria Tambahan**:
Melakukan Exploratory Data Analysis (EDA) secara bertahap:
1. Mengecek informasi pada dataset dengan fungsi `info()`, menghasilkan:
   - Terdapat 3 kolom bertipe object yaitu sex, smoker, region. Kolom ini termasuk kolom non-numerik
   - Terdapat 2 kolom bertipe integer yaitu age dan children
   - Terdapat 2 kolom bertipe float yaitu bmi dan charges

2. Mengecek deskripsi statistik dengan fungsi `describe()`, menghasilkan:
   - Indikasi atau logika dinilai seluruhnya masuk akal terhadap nilai numerik
   
3. Mengecek duplikasi data dengan fungsi `duplicated().sum()`, menghasilkan:
   - Jumlah duplikasi data berjumlah 1 baris dan dilakukan penghapusan data pada baris tersebut menggunakan fungsi `drop()`. Sehingga baris berjumlah 1.337

4. Mengecek missing value dengan fungsi `isna().sum()`, menghasilkan tidak terindikasinya nilai yang hilang di setiap fitur yang ada

5. Mendeteksi outlier dengan `boxplot`, menghasilkan:
   - Fitur age tidak terdapat outlier
   - Fitur BMI terdapat outlier, tetapi hal ini dianggap normal, sebab nilai BMI masih masuk akal direntang 10-60
   - Fitur charges terdapat outlier, tetapi hal tidak akan ditangani, sebab kita menginginkan model memprediksi nilai ekstrem, sebagai bahan prediksi perusahaan asuransi dengan berbagai faktor yang ada

6. Melakukan univariate analysis terhadap masing-masing fitur dengan `countplot` dan `histogram`, menghasilkan:
   a. Categorical features
      - Grafik fitur sex menunjukkan bahwa jenis kelamin laki-laki dan perempuan pada dataset hampir seimbang di angka 50.5% (675 sampel) dan 49.5% (662 sampel)
      - Grafik fitur smoker menunjukkan bahwa 20% pelanggan/pasien merupakan perokok. Selebihnya merupakan status perokok tidak aktif. Hal ini menandakan bahwa lebih dari setengah pelanggan/pasien dari perusahaan asuransi bukan perokok
      - Grafik fitur region menunjukkan bahwa pelanggan/pasien terbanyak berasal dari region southeast dengan 27.2%, selebihnya sama rata berasal dari region lain 
   b. Numerical features
      - Rentang usia terbilang cukup luas, dengan konsentrasi yang lebih tinggi pada usia muda
      - Distribusi BMI cenderung mengikuti pola normal dengan sebagian besar nilai berada di rentang yang umum
      - Sebagian besar pelanggan/pasien tidak memiliki anak
      - Biaya medis yang dikenakan sangat bervariasi, dengan sebagian besar berada di tingkat yang lebih rendah dan sebagian kecil dengan biaya yang jauh lebih tinggi

  7. Melakukan multivariate analysis untuk menilai relasi antar fitur terhadap fitur target (charges)
     a. Categorical features (menggunakan `catplot`)
        - Fitur sex memiliki pengaruh atau dampak yang kecil terhadap rata-rata biaya medis
        - Fitur smoker, memiliki pengaruh atau dampak yang besar terhadap rata-rata biaya medis.
        - Fitur region memiliki pengaruh atau dampak yang kecil terhadap rata-rata biaya medis.
     b. Numerical features (menggunakan `pairplot` dan `heatmap`)
        - Fitur age (0.30) dan bmi (0.20) memiliki skor korelasi yang terindikasi positif dengan fitur target charges
        - Fitur children memiliki korelasi yang sangat kecil (0.07). Sehingga, fitur tersebut dapat di-drop.

## Data Preparation
Melakukan tiga tahap persiapan data, yaitu:
1. Encoding fitur kategori
2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
3. Standarisasi.

**Rubrik/Kriteria Tambahan**:
1. Encoding fitur kategori menggunakan `OneHotEncoding` dan `LabelEncoder`. Hal ini dilakukan sebab model regresi membutuhkan input numerik, maka dari itu fitur kategori yang bertipe object di rubah menjadi numerik agar model mengenali data kategorikal
2. Membagi dataset menjadi data train dan data test dengan perbandingan 80:20 menggunakan `train_test_split`, sebab data yang kita miliki berada di kisaran 1000an sampel sehingga ini menjadi ideal. Hal ini dilakukan guna melakukan tahap training pada model menggunakan data train, lalu melakukan tahap evaluasi menggunakan data test
3. Standarisasi (scaling) menggunakan `StandardScaler` terhadap data yang telah displit sebelumnya. `StandardScaler` menghasilkan distribusi angka rentang 1,0,-1. Hal ini dilakukan agar algoritma stidak terpengaruh oleh perbedaan skala antar fitur.

## Modeling
Pada tahap ini, tahap pengembangan model machine learning dilakukan dengan menggunakan tiga algoritma yaitu Linear Regression, Random Forest Regressor, dan Gradien Boosting Regressor. Kemudian, akan dievaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:
1. Linear Regression
   Algoritma dasar dalam regresi yang mencoba mencari hubungan linear antara fitur dan target
   **Paramater yang digunakan**
   - Default dari `LinearRegression()`
   
   **Hasil Evaluasi**
   - MAE: 4141.22
   - MSE: 38,156,957.89
   - R²: 0.79

   **Kelebihan**
   - Sederhana dan cepat untuk dilatih
   - Mudah untuk diinterpretasikan

   **Kekurangan**
   - Tidak mampu menangkap hubungan non-linear.
   - Sensitif terhadap outlier.

2. Random Forest Regressor
   Metode ensemble berbasis pohon keputusan yang menggabungkan banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting
   **Paramater yang digunakan**
   - `n_estimator=150`: jumlah pohon dalam hutan
   - `max_depth=15`: kedalaman maksimum setiap pohon
   - `random_state=123`: mengontrol random number generator yang digunakan
   - `n_jobs=-1`: memastikan semua proses berjalan secara paralel
   
   **Hasil Evaluasi**
   - MAE: 2705.10
   - MSE: 22,177,523.66
   - R²: 0.88

   **Kelebihan**
   - Mampu menangani data kompleks dan hubungan non-linear
   - Tidak mudah overfitting karena menggunakan banyak pohon

   **Kekurangan**
   - Model cukup kompleks dan sulit untuk diinterpretasi
   - Waktu pelatihan lebih lama dibandingkan Linear Regression

3. Gradient Boosting Regressor
   Metode boosting yang membangun model secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya.
   **Paramater yang digunakan**
   - `n_estimator=150`: jumlah boosting stage
   - `max_depth=15`: kedalaman maksimum setiap pohon
   - `learning_rate=0.05`: seberapa besar kontribusi setiap pohon terhadap model akhir
   - `random_state=42`: mengontrol random number generator yang digunakan
   
   **Hasil Evaluasi**
   - MAE: 2403.83
   - MSE: 18,097,718.56
   - R²: 0.90

   **Kelebihan**
   - Sering memberikan akurasi prediksi yang tinggi
   - Dapat menangani kompleksitas hubungan dalam data

   **Kekurangan**
   - Waktu pelatihan lebih lama dari Linear Regression
   - Rentan terhadap overfitting

**Pemilihan model terbaik**
Berdasarkan hasil evaluasi pada data uji, model dengan performa terbaik dipilih dengan mempertimbangkan nilai MAE terendah, MSE terendah, dan R² tertinggi.
Jika dibandingkan, Gradient Boosting Regressor menunjukkan performa terbaik dengan nilai MAE paling rendah, MSE paling rendah, R² paling tinggi

Oleh karena itu, Gradient Boosting Regressor dipilih sebagai model terbaik karena:
- Memiliki kemampuan menangkap hubungan kompleks dan non-linear antar fitur
- Memberikan prediksi yang sangat akurat, menjadikannya sangat cocok ketika ketepatan prediksi menjadi prioritas utama.
- Lebih baik dibandingkan Random Forest dan Linear Regression dalam hal generalisasi pada data ini

## Evaluation
Untuk mengevaluasi performa model regresi yang dibangun, digunakan tiga metrik utama, yaitu Mean Absolute Error (MAE), Mean Squared Error (MSE), R-Squared (R²)
**1. Mean Absolute Error (MAE)**
MAE mengukur rata-rata selisih absolut antara nilai sebenarnya dengan hasil prediksi. Semakin kecil MAE, maka model semakin akurat.
Formula:


Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

