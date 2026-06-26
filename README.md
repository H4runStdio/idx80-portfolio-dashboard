# IDX80 Portfolio Intelligence Dashboard

Implementasi XGBoost dan Black-Litterman Model untuk Optimasi Portofolio Saham pada Indeks IDX80.

## Struktur Proyek

```
idx80_dashboard/
├── app.py                          # Entry point Streamlit
├── requirements.txt                # Dependensi Python
├── packages.txt                    # Dependensi sistem (wajib untuk xgboost)
├── run_with_ngrok.py               # Script bantu jalankan lokal via ngrok (opsional)
├── data/
│   └── IDX80_5yr.xlsx              # Data harga & volume 80 saham
├── utils/
│   ├── data_loader.py              # Pemuatan & caching data
│   ├── preprocessing.py            # Cleaning & perhitungan return
│   ├── xgboost_model.py            # Feature engineering, training, forecasting
│   ├── black_litterman.py          # Model Black-Litterman & optimasi
│   ├── risk_metrics.py             # VaR, CVaR, risk contribution, stress test
│   └── styling.py                  # CSS, Bootstrap Icons, komponen UI
├── sections/
│   ├── section1_overview.py        # Pemilihan saham & ringkasan
│   ├── section2_prediction.py      # Detail prediksi XGBoost
│   ├── section3_optimization.py    # Detail optimasi Black-Litterman
│   ├── section4_simulation.py      # Simulasi what-if & manajemen risiko
│   └── section5_about.py           # Informasi penyusun
├── assets/
│   └── style.css
└── .streamlit/
    └── config.toml
```

## Deploy ke Streamlit Community Cloud

Lihat panduan langkah-demi-langkah lengkap pada percakapan terkait, atau ringkasannya:

1. Push seluruh folder ini ke repository GitHub publik.
2. Buka [share.streamlit.io](https://share.streamlit.io), masuk dengan GitHub.
3. Klik "Create app", pilih repo, branch `main`, dan file utama `app.py`.
4. Klik "Deploy" dan tunggu proses build selesai.

File `packages.txt` (berisi `libgomp1`) **wajib ada** di root repo agar XGBoost dapat berjalan di Streamlit Cloud — tanpa file ini, app akan crash saat import tanpa pesan error yang jelas.

## Menjalankan Secara Lokal

```bash
cd idx80_dashboard
pip install -r requirements.txt
streamlit run app.py
```

Dashboard akan terbuka otomatis di `http://localhost:8501`.

## Menjalankan dengan ngrok (opsional, untuk akses publik dari lokal/Colab)

Gunakan opsi ini hanya jika Anda menjalankan dari komputer lokal atau Google Colab dan ingin tunnel publik sementara tanpa deploy permanen.

```bash
pip install pyngrok
python run_with_ngrok.py --token NGROK_AUTHTOKEN_ANDA
```

## Catatan Teknis

- **Data**: 80 saham konstituen IDX80, harga penutupan & volume harian, 8 Juni 2021 – 5 Juni 2026. Beberapa saham (GOTO, MTEL, PGEO, MBMA) memiliki rentang historis lebih pendek karena listing belakangan; sistem otomatis menyesuaikan melalui inner join tanggal saat menyusun matriks harga untuk kombinasi saham apa pun yang dipilih. Akibatnya, jika salah satu saham terpilih memiliki histori lebih pendek (misalnya GOTO), seluruh saham lain dalam portofolio tersebut juga akan dilatih hanya menggunakan rentang tanggal yang sama (bukan histori penuhnya), demi menjaga konsistensi periode antar model XGBoost dan matriks kovarians pada Black-Litterman.
- **XGBoost**: target prediksi adalah log-return harian (bukan harga absolut) agar lebih stasioner antar saham dengan skala harga berbeda. Forecast multi-hari dilakukan secara recursive.
- **Black-Litterman**: bobot pasar didekati equal-weight karena dataset tidak menyertakan data kapitalisasi pasar. Jika Anda memiliki data kapitalisasi pasar riil, ini dapat disuntikkan melalui parameter `market_weights` pada `run_black_litterman()` di `utils/black_litterman.py`.
- **Performa**: melatih 10 model XGBoost (satu per saham) berjalan beberapa detik hingga puluhan detik tergantung spesifikasi mesin. Hasil pipeline disimpan di `st.session_state` agar berpindah antar section tidak memicu pelatihan ulang.

## Lisensi Data dan Penggunaan

Dashboard ini dibangun sebagai proyek portofolio pribadi. Sesuaikan Section 5 (Informasi Penyusun) dengan identitas dan sumber data Anda sebelum dipublikasikan.
