# Analisis Sentimen Ulasan Produk dengan IndoBERT, SVM, dan VADER

Proyek ini bertujuan untuk menganalisis sentimen dari ulasan produk berbahasa Indonesia menggunakan tiga pendekatan yang berbeda:
1.  **Fine-Tuning Model Transformer**: Menggunakan model **IndoBERT** yang telah di-pre-train.
2.  **Machine Learning Klasik**: Menggunakan **Support Vector Machine (LinearSVC)** dengan fitur TF-IDF.
3.  **Metode Berbasis Leksikon**: Menggunakan **VADER** yang diperkaya dengan leksikon sentimen Bahasa Indonesia (InSet).

Tujuannya adalah untuk membandingkan performa, kecepatan, dan kompleksitas dari setiap metode dalam tugas klasifikasi sentimen biner (Positif vs. Negatif).

## Dataset

Dataset yang digunakan adalah `20191002-reviews.csv`, yang berisi ulasan produk dari salah satu platform e-commerce di Indonesia.

-   **Fitur Utama**: `rating` (peringkat bintang) dan `reviewContent` (teks ulasan).
-   **Label Sentimen**: Label sentimen dibuat berdasarkan kolom `rating`:
    -   **Positif**: Rating 4 atau 5.
    -   **Negatif**: Rating 1 atau 2.
-   **Catatan**: Ulasan dengan rating 3 (netral) tidak diikutsertakan dalam analisis untuk menjaga fokus pada klasifikasi biner yang jelas.

## Metodologi & Pendekatan

Tiga notebook terpisah digunakan untuk mengimplementasikan setiap pendekatan.

### 1. Fine-Tuning IndoBERT (`BERT_FINAL.ipynb`)

Pendekatan ini menggunakan model transformer canggih yang sudah dilatih secara ekstensif pada korpus Bahasa Indonesia.

-   **Model**: `indobenchmark/indobert-base-p1` dari Hugging Face.
-   **Pra-pemrosesan**:
    -   Data teks dibiarkan semirip mungkin dengan aslinya (tanpa *case folding* atau penghapusan tanda baca manual) karena model BERT mampu memahami konteks dari teks mentah.
    -   Teks diubah menjadi token numerik menggunakan `AutoTokenizer` dari Hugging Face.
    -   Dataset dibagi menjadi 80% data latih dan 20% data uji.
-   **Pelatihan**:
    -   Model di-*fine-tune* pada dataset ulasan selama 1 epoch.
    -   Proses ini sangat efektif namun memerlukan sumber daya komputasi yang signifikan (disarankan menggunakan GPU).

### 2. Support Vector Machine (LinearSVC) (`SVM_FINAL.ipynb`)

Ini adalah pendekatan machine learning klasik yang terkenal solid untuk tugas klasifikasi teks.

-   **Model**: `LinearSVC` dari Scikit-learn, yang merupakan implementasi SVM yang lebih cepat.
-   **Pra-pemrosesan & Feature Engineering**:
    -   **Pembersihan Teks**: Teks ulasan dibersihkan secara menyeluruh:
        -   *Case Folding*: Mengubah semua teks menjadi huruf kecil.
        -   Menghapus URL, tag HTML, dan karakter non-alfabet.
        -   Normalisasi spasi berlebih.
    -   **Vektorisasi**: Teks yang sudah bersih diubah menjadi representasi numerik menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF mengukur seberapa penting sebuah kata dalam dokumen relatif terhadap keseluruhan korpus.
-   **Pelatihan & Optimasi**:
    -   `Pipeline` dari Scikit-learn digunakan untuk menggabungkan langkah vektorisasi dan klasifikasi.
    -   `GridSearchCV` diterapkan untuk mencari hyperparameter terbaik, termasuk kombinasi n-gram (unigram & bigram) dan parameter regularisasi `C` dari SVM.

### 3. VADER dengan Leksikon Indonesia (`VADER_FINAL.ipynb`)

Pendekatan ini bersifat *rule-based* dan tidak memerlukan pelatihan model machine learning.

-   **Model**: `SentimentIntensityAnalyzer` dari library VADER.
-   **Pra-pemrosesan**:
    -   Pembersihan teks dasar dilakukan (case folding, penghapusan angka dan tanda baca).
-   **Analisis**:
    -   Kekuatan utama pendekatan ini adalah penggabungan leksikon eksternal. Leksikon VADER yang berbasis Bahasa Inggris diperbarui dengan **Leksikon Sentimen Indonesia (InSet)** untuk meningkatkan kemampuannya dalam memahami kata-kata positif dan negatif dalam Bahasa Indonesia.
    -   VADER menganalisis setiap kata dalam ulasan, mencocokkannya dengan leksikon, dan menghasilkan skor `compound` antara -1 (sangat negatif) dan 1 (sangat positif).
    -   Sentimen akhir ditentukan berdasarkan nilai skor `compound`.

## Hasil & Perbandingan

Berikut adalah perbandingan performa dari ketiga model pada data uji yang sama.

| Model                               | Akurasi | Presisi (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) | Catatan                                        |
| ----------------------------------- | :-----: | :-----------------: | :----------------: | :------------------: | ---------------------------------------------- |
| **LinearSVC (SVM) + TF-IDF** | **99%** |      **98%**     |     **97%**      |      **97%**       | Performa terbaik secara keseluruhan.           |
| **Fine-Tuning IndoBERT** |  97%    |         91%         |        91%         |         91%        | Sangat baik, sedikit di bawah SVM.             |
| **VADER + Leksikon Indonesia** |  43%    |         55%         |        64%         |         39%        | Performa paling rendah, kesulitan pada konteks.|

### Analisis Hasil:

-   **LinearSVC (SVM)** menunjukkan performa tertinggi. Ini membuktikan bahwa dengan pra-pemrosesan teks yang solid dan fitur TF-IDF yang baik, model machine learning klasik bisa sangat kompetitif dan bahkan unggul.
-   **IndoBERT** juga memberikan hasil yang sangat baik dengan akurasi 97%. Keunggulannya adalah tidak memerlukan pembersihan teks yang rumit karena model ini belajar representasi langsung dari data. Namun, proses *fine-tuning*-nya memerlukan waktu dan sumber daya komputasi yang lebih besar.
-   **VADER** yang diperkaya dengan leksikon Indonesia memiliki performa yang jauh di bawah kedua model lainnya. Meskipun memiliki recall yang tinggi untuk sentimen negatif (90%), recall untuk sentimen positif sangat rendah (38%), yang berarti banyak ulasan positif yang salah diklasifikasikan. Ini menyoroti keterbatasan metode berbasis leksikon yang kesulitan menangani konteks, sarkasme, dan struktur kalimat yang kompleks.

## Visualisasi

Setiap notebook menghasilkan *confusion matrix* untuk mengevaluasi performa model secara visual.

-   **Confusion Matrix IndoBERT**: Menunjukkan akurasi tinggi dengan sedikit kesalahan klasifikasi baik untuk kelas positif maupun negatif.
-   **Confusion Matrix SVM**: Menampilkan hasil yang hampir sempurna dengan jumlah *false positives* dan *false negatives* yang sangat minim.
-   **Confusion Matrix VADER**: Secara jelas menunjukkan ketidakseimbangan performa, di mana mayoritas kesalahan terjadi karena model salah mengklasifikasikan ulasan positif sebagai negatif.

Dibuat dengan ❤️ oleh **Azzam**.
