# Laporan Proyek Machine Learning - Rafif Idris Ardhana

## Domain Proyek: Kesehatan
Perusahaan asuransi kesehatan adalah lembaga yang menyediakan perlindungan finansial bagi individu terhadap biaya medis yang tidak terduga. Suatu perusahaan asuransi bertanggung jawab untuk menanggung sebagian atau seluruh biaya perawatan medis yang timbul akibat penyakit atau kecelakaan individu/kelompok. Bagi suatu perusahaan asuransi penting untuk memperkirakan kisaran biaya yang perlu dipersiapkan untuk individu/kelompok di masa depan. Perkiraan atau prediksi ini memungkinkan perusahaan asuransi menyesuaikan premi secara adil dan efisien.

**Rubrik/Kriteria Tambahan**:
- Masalah ini sangat penting bagi banyak pemangku kepentingan perusahaan kesehatan khususnya asuransi, sebab estimasi biaya yang akurat dapat membantu perusahaan untuk merencanakan masa depan dan memprioritaskan alokasi sumber daya manajemen. Masalah ini dapat diselesaikan dengan mengadopsi teknologi terkini yaitu Machine Learning. Machine Learning (ML) merupakan salah satu aspek kecerdasan komputasional yang dapat memecahkan berbagai masalah dalam berbagai aplikasi dan sistem dalam hal memanfaatkan data historis. Khususnya di sektor asuransi, ML dapat membantu meningkatkan efisiensi penyusunan kebijakan termasuk premi, klaim, dan lainnya. Algoritma ML sangat baik dalam memprediksi pengeluaran pasien yang kemungkinan berbiaya tinggi dan sangat membutuhkan.
- Referensi:
  ul Hassan, C. A., Iqbal, J., Hussain, S., AlSalman, H., Mosleh, M. A., & Sajid Ullah, S. (2021). A computational intelligence approach for predicting medical insurance cost. Mathematical Problems in Engineering, 2021
  Diakses melalui: https://onlinelibrary.wiley.com/doi/abs/10.1155/2021/1162553

## Business Understanding
Suatu perusahaan asuransi kesehatan memiliki tanggung jawab yang besar terhadap perjanjian medis yang dilakukan dengan individu/kelompok. Perusahaan akan mengalami kerugian signifikan jika salah menentukan premi asuransi. Misalnya premi yang terlalu rendah akan menyebabkan kerugian finansial saat klaim melonjak, sedangkan premi yang terlalu tinggi menyebabkan kehilangan pelanggan akibat nilai yang tidak kompetitif. Selain itu, perusahaan asuransi tersebut juga tidak dapat mengidentifikasi pasien dengan resiko tinggi yang berkemungkinan dapat ditawarkan program kesehatan khusus.

Oleh karena itu, penting bagi perusahaan asuransi kesehatan untuk mengetahui dan dapat memprediksi kejadian di masa depan dengan menggunakan suatu sistem. Hal ini akan berdampak positif pada penyusunan anggaran, perencanaan risiko, dan penyesuaian strategi akan mengandalkan teknologi atau automation judgement, yang mana dapat membantu pengambilan keputusan dan prosesnya tidak memakan waktu lama.

### Problem Statements
- Pernyataan Masalah 1: Dari serangkaian atribut(fitur) yang ada, atribut(fitur) apa yang paling berpengaruh terhadap biaya medis (charge) asuransi?
- Pernyataan Masalah 2: Bagaimana cara memprediksi biaya medis (charges) yang akan dikeluarkan seseorang berdasarkan fitur yang ada?

### Goals
- Tujuan 1: Mengetahui atribut(fitur) yang paling berkorelasi dengan biaya medis (charges)
- Tujuan 2: Membuat machine learning yang dapat memprediksi biaya medis (charges) berdasarkan fitur yang ada

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

**Rubrik/Kriteria Tambahan**:\
Melakukan Exploratory Data Analysis (EDA) secara bertahap:
1. Mengecek informasi pada dataset dengan fungsi `info()`\
   **Menghasilkan informasi:**
   - Terdapat 3 kolom bertipe object yaitu sex, smoker, region. Kolom ini termasuk kolom non-numerik
   - Terdapat 2 kolom bertipe integer yaitu age dan children
   - Terdapat 2 kolom bertipe float yaitu bmi dan charges

2. Mengecek deskripsi statistik dengan fungsi `describe()`\
   **Menghasilkan informasi:**
   - Indikasi atau logika dinilai seluruhnya masuk akal terhadap nilai numerik

3. Mendeteksi outlier dengan `boxplot`\
   **Menghasilkan informasi:**
   - Fitur age tidak terdapat outlier\
     Visualisasi:\
     ![Screenshot 2025-04-23 203225](https://github.com/user-attachments/assets/244da73f-38f0-4988-b4cd-1ea0019fdea1)
     
   - Fitur BMI terdapat outlier, tetapi hal ini dianggap normal, sebab nilai BMI masih masuk akal direntang 10-60\
     Visualisasi:\
     ![Screenshot 2025-04-23 203239](https://github.com/user-attachments/assets/791cfed9-debb-447e-876d-1eaae63237be)

   - Fitur charges terdapat outlier, tetapi hal tidak akan ditangani, sebab kita menginginkan model memprediksi nilai ekstrem, sebagai bahan prediksi perusahaan asuransi dengan berbagai faktor yang ada\
     Visualisasi:\
     ![Screenshot 2025-04-23 203253](https://github.com/user-attachments/assets/794433a2-9fa6-4711-83f1-20242b3d4d9c)

4. Melakukan univariate analysis terhadap masing-masing fitur dengan `countplot` dan `histogram`\
   **Menghasilkan informasi:**\
   a. Categorical features
      - Grafik fitur sex menunjukkan bahwa jenis kelamin laki-laki dan perempuan pada dataset hampir seimbang di angka 50.5% (675 sampel) dan 49.5% (662 sampel)\
        Visualisasi:\
        ![Screenshot 2025-04-23 203314](https://github.com/user-attachments/assets/d2e51245-aeb7-4cf1-8796-3c2e45fb12fd)

      - Grafik fitur smoker menunjukkan bahwa 20% pelanggan/pasien merupakan perokok. Selebihnya merupakan status perokok tidak aktif. Hal ini menandakan bahwa lebih dari setengah pelanggan/pasien dari perusahaan asuransi bukan perokok\
        Visualisasi:\
        ![Screenshot 2025-04-23 203328](https://github.com/user-attachments/assets/740f1fac-67de-4a06-9909-51983b6c28ac)

      - Grafik fitur region menunjukkan bahwa pelanggan/pasien terbanyak berasal dari region southeast dengan 27.2%, selebihnya sama rata berasal dari region lain\
        Visualisasi:\
        ![Screenshot 2025-04-23 203343](https://github.com/user-attachments/assets/55df7b29-441c-4085-8868-ca39a3a58ac9)

   b. Numerical features
      - Rentang usia terbilang cukup luas, dengan konsentrasi yang lebih tinggi pada usia muda
      - Distribusi BMI cenderung mengikuti pola normal dengan sebagian besar nilai berada di rentang yang umum
      - Sebagian besar pelanggan/pasien tidak memiliki anak
      - Biaya medis yang dikenakan sangat bervariasi, dengan sebagian besar berada di tingkat yang lebih rendah dan sebagian kecil dengan biaya yang jauh lebih tinggi\
        Visualisasi:\
        ![Screenshot 2025-04-23 203359](https://github.com/user-attachments/assets/23eadc29-1460-49aa-9654-023a2206c968)


5. Melakukan multivariate analysis untuk menilai relasi antar fitur terhadap fitur target (charges)\
   **Menghasilkan informasi:**\
   a. Categorical features (menggunakan `catplot`)\
      - Fitur sex memiliki pengaruh atau dampak yang kecil terhadap rata-rata biaya medis\
        Visualisasi:\
        ![Screenshot 2025-04-23 203421](https://github.com/user-attachments/assets/c44fede7-4953-405f-975b-7c97351d4ab7)

      - Fitur smoker, memiliki pengaruh atau dampak yang besar terhadap rata-rata biaya medis.\
        Visualisasi:\
        ![Screenshot 2025-04-23 203428](https://github.com/user-attachments/assets/ca5c9fd3-d0aa-4a73-9767-80ebc81f020c)

      - Fitur region memiliki pengaruh atau dampak yang kecil terhadap rata-rata biaya medis.\
        Visualisasi:\
        ![Screenshot 2025-04-23 203441](https://github.com/user-attachments/assets/63bba737-1b9a-4277-be72-ac639b07b4fe)

          
   b. Numerical features (menggunakan `pairplot` dan `heatmap`)\
      - Fitur age (0.30) dan bmi (0.20) memiliki skor korelasi yang terindikasi positif dengan fitur target charges
      - Fitur children memiliki korelasi yang sangat kecil (0.07). Sehingga, fitur tersebut dapat di-drop.\
        Visualisasi:\
        ![Screenshot 2025-04-23 203502](https://github.com/user-attachments/assets/4e576345-d8be-4adb-9892-7b9ffa34aaf1)


## Data Preparation
Melakukan lima tahap persiapan data, yaitu:
1. Menangani Duplikasi Data
2. Mengecek Missing Value
3. Seleksi Fitur
4. Encoding fitur kategori
5. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
6. Standarisasi.

**Rubrik/Kriteria Tambahan**:
1. Mengecek duplikasi data dengan fungsi `duplicated().sum()`\
   - Jumlah duplikasi data berjumlah 1 baris dan dilakukan penghapusan data pada baris tersebut menggunakan fungsi `drop()`. Sehingga baris berjumlah 1.337
2. Mengecek missing value dengan fungsi `isna().sum()`\
   - Tidak terindikasinya nilai yang hilang di setiap fitur yang ada
3. Seleksi fitur dengan melakukan drop pada fitur children dengan fungsi `drop()`
   - Fitur children memiliki korelasi rendah terhadap fitur biaya medis (charges), sehingga fitur ini dapat dihapus
4. Encoding fitur kategori menggunakan `OneHotEncoding` dan `LabelEncoder`\
   - Hal ini dilakukan sebab model regresi membutuhkan input numerik, maka dari itu fitur kategori yang bertipe object di rubah menjadi numerik agar model mengenali data kategorikal
5. Membagi dataset menjadi data train dan data test dengan perbandingan 80:20 menggunakan `train_test_split`\
   - Data yang kita miliki berada di kisaran 1000an sampel sehingga ini menjadi ideal. Hal ini dilakukan guna melakukan tahap training pada model menggunakan data train, lalu melakukan tahap evaluasi menggunakan data test
6. Standarisasi (scaling) menggunakan `StandardScaler` terhadap data yang telah displit sebelumnya\
   - `StandardScaler` menghasilkan distribusi angka rentang 1,0,-1. Hal ini dilakukan agar algoritma stidak terpengaruh oleh perbedaan skala antar fitur.

## Modeling
Pada tahap ini, tahap pengembangan model machine learning dilakukan dengan menggunakan tiga algoritma yaitu Linear Regression, Random Forest Regressor, dan Gradien Boosting Regressor. Kemudian, akan dievaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:\
**1. Linear Regression**\
   Algoritma dasar dalam regresi yang mencoba mencari hubungan linear antara fitur dan target
   
   **Paramater yang digunakan**
   - Default dari `LinearRegression()`

   **Kelebihan**
   - Sederhana dan cepat untuk dilatih
   - Mudah untuk diinterpretasikan

   **Kekurangan**
   - Tidak mampu menangkap hubungan non-linear.
   - Sensitif terhadap outlier.

**2. Random Forest Regressor**\
   Metode ensemble berbasis pohon keputusan yang menggabungkan banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting
   
   **Paramater yang digunakan**
   - `n_estimator=150`: jumlah pohon dalam hutan
   - `max_depth=15`: kedalaman maksimum setiap pohon
   - `random_state=123`: mengontrol random number generator yang digunakan
   - `n_jobs=-1`: memastikan semua proses berjalan secara paralel

   **Kelebihan**
   - Mampu menangani data kompleks dan hubungan non-linear
   - Tidak mudah overfitting karena menggunakan banyak pohon

   **Kekurangan**
   - Model cukup kompleks dan sulit untuk diinterpretasi
   - Waktu pelatihan lebih lama dibandingkan Linear Regression

**3. Gradient Boosting Regressor**\
   Metode boosting yang membangun model secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya.
   
   **Paramater yang digunakan**
   - `n_estimator=150`: jumlah boosting stage
   - `max_depth=3`: kedalaman maksimum setiap pohon
   - `learning_rate=0.05`: seberapa besar kontribusi setiap pohon terhadap model akhir
   - `random_state=42`: mengontrol random number generator yang digunakan

   **Kelebihan**
   - Sering memberikan akurasi prediksi yang tinggi
   - Dapat menangani kompleksitas hubungan dalam data

   **Kekurangan**
   - Waktu pelatihan lebih lama dari Linear Regression
   - Rentan terhadap overfitting

**Pemilihan model terbaik**\
Berdasarkan hasil evaluasi pada data uji, model dengan performa terbaik dipilih dengan mempertimbangkan nilai MAE terendah, MSE terendah, dan R² tertinggi.\
Jika dibandingkan, **Gradient Boosting Regressor** menunjukkan performa terbaik dengan nilai MAE paling rendah, MSE paling rendah, R² paling tinggi

Oleh karena itu, **Gradient Boosting Regressor** dipilih sebagai model terbaik karena:
- Memiliki kemampuan menangkap hubungan kompleks dan non-linear antar fitur
- Memberikan prediksi yang sangat akurat, menjadikannya sangat cocok ketika ketepatan prediksi menjadi prioritas utama.
- Lebih baik dibandingkan Random Forest dan Linear Regression dalam hal generalisasi pada data ini

## Evaluation
**Evaluasi Numerik**\
Untuk mengevaluasi performa model regresi yang dibangun, digunakan tiga metrik utama, yaitu Mean Absolute Error (MAE), Mean Squared Error (MSE), R-Squared (R²)\
**1. Mean Absolute Error (MAE)**\
MAE mengukur rata-rata selisih absolut antara nilai sebenarnya dengan hasil prediksi. Semakin kecil MAE, maka model semakin akurat.\
Formula:

![MAE Formula](https://github.com/user-attachments/assets/fb6a4fff-0e05-4a2c-99a7-e4e5549f87f1)

- yi = nilai aktual
- 𝑦^𝑖 = nilai prediksi
- n = jumlah total sampel

MAE menghitung rata-rata dari selisih absolut antara nilai aktual dan hasil prediksi. Misalnya jika model memprediksi biaya medis sebesar 12.000 tetapi aslinya 10.000, maka selisihnya 2.000. Jika banyak data yang seperti ini, MAE akan menunjukkan rata-rata seberapa jauh prediksi dari kenyataan

**2. Mean Squared Error (MSE)**\
MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan prediksi. Karena nilai selisih dikuadratkan, MSE memberikan penalti lebih besar terhadap error yang besar.\
Formula:

![MSE_Formula](https://github.com/user-attachments/assets/b8d3c59e-b3e9-4fcc-b6d5-0f79dcb9ae6b)

Mengukur rata-rata selisih kuadrat antara nilai aktual dan prediksi. Berbeda dengan MAE, karena error-nya dikuadratkan, kesalahan yang besar akan diberi penalti lebih besar. Misalnya hasil selisih 2.000 maka dikuadratkan menjadi 4.000.000. Maka, model yang sering membuat error besar akan terlihat jelas dari nilai MSE yang sangat besar.

**3. R-Squared (R²)**\
R² mengukur seberapa besar variasi dalam data yang dapat dijelaskan oleh model. Nilai R² berada dalam rentang 0 hingga 1. Semakin mendekati 1, semakin baik model menjelaskan data.\
Formula:

![R_Squared_Formula](https://github.com/user-attachments/assets/e047e1ed-208c-4474-abe0-3caf55c903f9)

- SS<sub>res</sub> = jumlah kuadrat residual (kesalahan prediksi)
- 𝑆𝑆<sub>𝑡𝑜𝑡</sub> = total variasi dari data terhadap rata-rata
  
R² mengukur seberapa besar variansi (penyebaran data) yang bisa dijelaskan oleh model.
- R² = 1 → prediksi sempurna
- R² = 0 → model tidak lebih baik dari sekadar menebak rata-rata
- R² < 0 → model lebih buruk dari model yang hanya menebak rata-rata

**Hasil Evaluasi Model**\
Berikut adalah perbandingan hasil evaluasi dari tiga algoritma regresi yang digunakan:
| Model              | MAE       | MSE              | R²   |
|--------------------|-----------|------------------|------|
| Linear Regression  | 4141.22   | 38,156,957.89    | 0.79 |
| Random Forest      | 2705.10   | 22,177,523.66    | 0.88 |
| Gradient Boosting  | 2403.83   | 18,097,718.56    | 0.90 |

Dari hasil tersebut, dapat disimpulkan bahwa **Gradient Boosting Regressor** menunjukkan performa terbaik karena memiliki:
- Nilai MAE paling rendah, artinya rata-rata kesalahan prediksinya paling kecil
- Nilai MSE paling rendah, menunjukkan prediksi model relatif konsisten dan tidak terlalu jauh dari nilai aktual
- Nilai R² tertinggi (0.90), yang berarti model mampu menjelaskan 90% variasi dari target

**Evaluasi Deskriptif**\
Perusahaan asuransi kesehatan harus mampu mencegah beberapa kerugian yang dianggap merugikan perusahaan seperti menghindari kerugian finansial akibat premi yang salah ditentukan (terlalu rendah atau terlalu tinggi), mengidentifikasi individu berisiko tinggi, sehingga dapat ditawarkan program kesehatan preventif, merencanakan anggaran dan strategi keuangan secara tepat, berbasis prediksi yang akurat terhadap biaya medis. Maka dari itu, perusahaan asuransi kesehatan perlu memanfaatkan teknologi (ML) untuk pengambilan keputusan yang lebih cepat dan efisien dengan menggunakan algoritma yang performanya dianggap baik bagi kasus perusahaan. 

1. **Evaluasi terhadap Problem Statement**
    - **Dari serangkaian atribut(fitur) yang ada, atribut(fitur) apa yang paling berpengaruh terhadap biaya medis (charge) asuransi?**\
      Masalah ini telah berhasil dijawab melalui proses eksplorasi data dan analisis feature importance. Fitur-fitur yang memiliki pengaruh terhadap biaya medis akan dijawab di bagian Evaluasi Goals.Pengetahuan ini memungkinkan perusahaan untuk fokus pada faktor-faktor utama dalam penetapan premi asuransi.\
    - **Bagaimana cara memprediksi biaya medis (charges) yang akan dikeluarkan seseorang berdasarkan fitur yang ada?**\
      Masalah ini telah berhasil dijawab dengan membangun tiga model regresi, yaitu Linear Regression (baseline), Random Forest Regressor, dan Gradient Boosting Regressor, perusahaan kini dapat memprediksi biaya medis secara lebih akurat. Gradient Boosting muncul sebagai model terbaik, memberikan prediksi yang dapat diandalkan dalam pengambilan keputusan terkait premi dan pengelolaan risiko.

2. **Evaluasi terhadap Goals**
    - **Mengetahui atribut yang paling berkorelasi dengan biaya medis (charges)**\
      Tujuan ini tercapai melalui analisis korelasi dan feature importance dari model, sehingga perusahaan dapat mengenali faktor risiko utama. Fitur-fitur seperti usia (age), indeks massa tubuh (BMI), dan status merokok (smoker) terbukti memiliki pengaruh paling besar terhadap biaya medis seseorang.
    - **Membuat machine learning yang dapat memprediksi biaya medis (charges) berdasarkan fitur yang ada**\
      Tujuan ini tercapai melalui pembangunan model machine learning, kemudian dibandingkan berdasarkan metrik evaluasi. Model Gradient Boosting memberikan performa terbaik dengan R² sebesar 0.90, MAE terendah, dan MSE paling kecil. Model ini dapat digunakan perusahaan asuransi untuk memprediksi kebutuhan perusahaan di masa depan
      
3. **Evaluasi terhadap Solution Statements**
   - **Membangun model baseline menggunakan Linear Regression**\
     Model baseline berhasil dibangun dan memberikan dampak untuk tolok ukur awal performa model-model lainnya
   - **Membangun model alternatif menggunakan Random Forest Regressor dan Gradient Boosting Regressor**\
     Kedua model berhasil dikembangkan dan memberikan dampak performa yang lebih baik daripada model baseline, khususnya Gradient Boosting.
   - **Mengukur performa model menggunakan MAE, MSE, dan R² Score**\
     Berhasil melakukan proses evaluasi yang dilakukan dengan tiga metrik utama. Hal ini berdampak pada hasil yang diperoleh yaitu lebih objektif dan menyeluruh.
   - **Memilih model terbaik berdasarkan evaluasi metrik**\
     Model Gradient Boosting dipilih sebagai model final karena memiliki MAE dan MSE paling rendah serta R² tertinggi. Pemilihan model terbaik ini berdampak pada tingkat kemampuan prediksi machine learning yang akan digunakan perusahaan asuransi kesehatan

**Kesimpulan Evaluasi:**\
Dengan menggunakan model dan hasil yang telah dirancang terutama model **Gradient Boosting Regressor**, diharapkan perusahaan dapat mencegah resiko kerugian, seperti:
- Prediksi biaya medis yang akurat dapat membantu merencanakan premi yang adil dan kompetitif
- Kemampuan identifikasi risiko tinggi menjadi lebih baik untuk memungkinkan pemberian program kesehatan khusus pada individu tertentu
- Optimalisasi perencanaan keuangan dan pengendalian klaim, berkontribusi pada efisiensi manajemen risiko perusahaan
- Pemanfaatan teknologi mendukung pengambilan keputusan yang cepat dan berorientasikan data
