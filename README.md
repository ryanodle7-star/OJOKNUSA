# OJOKNUSA AI

Dashboard prototype berbasis Streamlit untuk eksplorasi perubahan garis pantai dan analisis awal risiko abrasi di Magepanda.

## Isi repository

- `app.py` — aplikasi Streamlit lengkap.
- `model/random_forest_magepanda.pkl` — model Random Forest Regressor yang direkonstruksi dari dataset proyek.
- `data/raw/Magepanda_synthetic_dataset_corrected.csv` — dataset sintetis/purwarupa.
- `requirements.txt` — dependensi aplikasi.

## Model

Model menggunakan 7 fitur:

`year`, `latitude`, `longitude`, `wave_height_m`, `wind_speed_ms`, `elevation_m`, `distance_to_road_m`

Target: `change_rate_m_year`.

Konfigurasi rekonstruksi: `RandomForestRegressor(n_estimators=100, random_state=42)`.

## Catatan penelitian

Dataset pada repository ini bersifat **sintetis/purwarupa**, bukan data observasi lapangan atau hasil penginderaan jauh aktual. Kategori risiko juga merupakan aturan prototype dan belum merupakan indeks risiko ilmiah tervalidasi. Hasil perlu divalidasi menggunakan data aktual sebelum digunakan untuk keputusan infrastruktur.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Hubungkan repository GitHub ini ke Streamlit Community Cloud dan pilih `app.py` sebagai entry point.
