# 📊 Dashboard Prediksi Harga Bitcoin - GRU vs LSTM

Dashboard interaktif untuk membandingkan performa model **GRU** dan **LSTM** dalam memprediksi harga Bitcoin (BTC-USD).

## 📋 Fitur Dashboard

| Row | Deskripsi |
|-----|-----------|
| **Row 1** | Riwayat harga Bitcoin (full history chart) |
| **Row 2** | Prediksi GRU vs Harga Aktual (kiri) & Prediksi LSTM vs Harga Aktual (kanan) |
| **Row 3** | Analisis detail hasil Test Set - GRU (kiri) & LSTM (kanan) |
| **Row 4** | Perbandingan metrik evaluasi Test Set: RMSE, MAE, MAPE, Akurasi |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualisasi**: Plotly
- **Data**: Yahoo Finance (yfinance)
- **Model**: TensorFlow / Keras (GRU & LSTM)
- **Preprocessing**: scikit-learn, ta (Technical Analysis)

## 📁 Struktur Project

```
Project/
├── app.py                          # Dashboard Streamlit
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── Model/
│   ├── gru_bitcoin_96%.keras       # Model GRU
│   └── lstm_bitcoin_88%.keras      # Model LSTM
├── TA_Model_GRU.ipynb              # Notebook training GRU
└── TA_Model_LSTM_NEW.ipynb         # Notebook training LSTM
```

## 🚀 Cara Running

### 1. Clone / Download Project

```bash
cd "path/to/Project"
```

### 2. Buat Virtual Environment (opsional tapi disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`.

## ⚙️ Konfigurasi Model

| Parameter | Nilai |
|-----------|-------|
| Data Source | Yahoo Finance (BTC-USD) |
| Periode Data | 2015-01-01 s/d sekarang |
| Lookback Window | 30 hari |
| Train Ratio | 70% |
| Validation Ratio | 15% |
| Test Ratio | 15% |
| Fitur Input | Close, Volume, RSI, MACD, MACD_Signal, MACD_Hist, Day_of_Week, Month |
| Normalisasi | MinMaxScaler (0-1) |

## 📊 Metrik Evaluasi

- **RMSE (USD)**: Root Mean Squared Error
- **MAE (USD)**: Mean Absolute Error
- **MAPE (%)**: Mean Absolute Percentage Error
- **Akurasi (%)**: 100% - MAPE

## 📝 Catatan

- Data Bitcoin diunduh secara real-time dari Yahoo Finance saat dashboard dijalankan
- Model sudah di-training sebelumnya dan disimpan di folder `Model/`
- Dashboard menggunakan caching untuk mempercepat loading
