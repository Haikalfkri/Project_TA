"""
Dashboard Perbandingan Model GRU vs LSTM - Prediksi Harga Bitcoin
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import tensorflow as tf
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard Prediksi Bitcoin - GRU vs LSTM",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    h1, h2, h3 {
        color: #f5f5f5 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30,30,60,0.9), rgba(20,20,50,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }
    .metric-label {
        font-size: 13px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value-gru {
        background: linear-gradient(90deg, #f7931a, #ffb347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value-lstm {
        background: linear-gradient(90deg, #8e44ad, #c39bd3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-title {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .header-title h1 {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f7931a, #ffb347, #64ffda);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-title p {
        color: #8892b0;
        font-size: 1rem;
        margin-top: 8px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ccd6f6 !important;
        padding: 16px 0 8px 0;
        border-bottom: 2px solid rgba(100,255,218,0.3);
        margin-bottom: 16px;
    }

    div[data-testid="stMetric"] {
        background: rgba(30,30,60,0.8);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 16px;
    }

    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING & PROCESSING (cached)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_data():
    """Download Bitcoin data, save to data/ folder, and compute technical indicators."""
    START_DATE = '2015-01-01'
    END_DATE = datetime.today().strftime('%Y-%m-%d')

    # Buat folder data jika belum ada
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    df_raw = yf.download('BTC-USD', start=START_DATE, end=END_DATE, progress=False)
    df_raw = df_raw.reset_index()

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [col[0] if col[1] == '' else col[0] for col in df_raw.columns]

    # Simpan data history mentah ke folder data
    raw_csv_path = os.path.join(data_dir, 'bitcoin_history_raw.csv')
    df_raw.to_csv(raw_csv_path, index=False)

    df = df_raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna()

    # Technical Indicators
    df['MA7'] = SMAIndicator(close=df['Close'], window=7).sma_indicator()
    df['MA21'] = SMAIndicator(close=df['Close'], window=21).sma_indicator()
    df['MA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.dropna().reset_index(drop=True)

    # Simpan data dengan indikator teknikal ke folder data
    processed_csv_path = os.path.join(data_dir, 'bitcoin_history_processed.csv')
    df.to_csv(processed_csv_path, index=False)

    return df_raw, df


@st.cache_resource(show_spinner=False)
def load_models():
    """Load saved GRU and LSTM models."""
    model_dir = os.path.join(os.path.dirname(__file__), 'Model')
    gru_path = os.path.join(model_dir, 'gru_model.keras')
    lstm_path = os.path.join(model_dir, 'lstm_model.keras')

    gru_model = tf.keras.models.load_model(gru_path)
    lstm_model = tf.keras.models.load_model(lstm_path)
    return gru_model, lstm_model


def prepare_data_for_prediction(df):
    """Prepare sequences and scalers matching notebook pipeline."""
    FEATURES = ['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Day_of_Week', 'Month']
    TARGET = 'Close'
    LOOKBACK = 30
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    data = df[FEATURES].values

    n_total_raw = len(data)
    n_train_raw = int((n_total_raw - LOOKBACK) * TRAIN_RATIO) + LOOKBACK

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(data[:n_train_raw])
    scaled_data = scaler_X.transform(data)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(df[[TARGET]].values[:n_train_raw])
    scaled_close = scaler_y.transform(df[[TARGET]].values)

    # Create sequences
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i - LOOKBACK:i])
        y.append(scaled_close[i, 0])
    X, y = np.array(X), np.array(y)

    n_total = len(X)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    dates = df['Date'].values[LOOKBACK:]
    dates_train = dates[:n_train]
    dates_val = dates[n_train:n_train + n_val]
    dates_test = dates[n_train + n_val:]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'dates_train': dates_train, 'dates_val': dates_val, 'dates_test': dates_test,
        'scaler_y': scaler_y
    }


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, MAPE, and Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    accuracy = max(0, 100 - mape)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Accuracy': accuracy}


def get_predictions(model, data_dict):
    """Get predictions for all sets and inverse transform."""
    scaler_y = data_dict['scaler_y']

    pred_train = model.predict(data_dict['X_train'], verbose=0)
    pred_val = model.predict(data_dict['X_val'], verbose=0)
    pred_test = model.predict(data_dict['X_test'], verbose=0)

    y_pred_train = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
    y_pred_val = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
    y_pred_test = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()

    y_true_train = scaler_y.inverse_transform(data_dict['y_train'].reshape(-1, 1)).flatten()
    y_true_val = scaler_y.inverse_transform(data_dict['y_val'].reshape(-1, 1)).flatten()
    y_true_test = scaler_y.inverse_transform(data_dict['y_test'].reshape(-1, 1)).flatten()

    return {
        'y_pred_train': y_pred_train, 'y_pred_val': y_pred_val, 'y_pred_test': y_pred_test,
        'y_true_train': y_true_train, 'y_true_val': y_true_val, 'y_true_test': y_true_test
    }


# ============================================================
# CHART HELPERS
# ============================================================
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,15,40,0.6)',
    font=dict(family='Inter', color='#ccd6f6'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', showline=False),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
    hovermode='x unified'
)


def create_history_chart(df_raw):
    """Create full Bitcoin history chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_raw['Date'], y=df_raw['Close'],
        mode='lines',
        name='Harga Penutupan (USD)',
        line=dict(color='#f7931a', width=2),
        fill='tozeroy',
        fillcolor='rgba(247,147,26,0.08)',
        hovertemplate='%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text='📈 Riwayat Harga Bitcoin (BTC-USD)', font=dict(size=20, color='#f5f5f5')),
        yaxis_title='Harga (USD)',
        xaxis_title='Tanggal',
        height=420,
        yaxis_tickformat='$,.0f'
    )
    return fig


def create_prediction_chart(dates_test, y_true, y_pred, model_name, color_actual, color_pred):
    """Create prediction vs actual chart."""
    dates_pd = pd.to_datetime(dates_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_pd, y=y_true, mode='lines', name='Harga Aktual',
        line=dict(color=color_actual, width=2),
        hovertemplate='%{x|%d %b %Y}<br>Aktual: $%{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_pd, y=y_pred, mode='lines', name='Prediksi ' + model_name,
        line=dict(color=color_pred, width=2, dash='dash'),
        hovertemplate='%{x|%d %b %Y}<br>Prediksi: $%{y:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f'Prediksi {model_name} vs Harga Aktual', font=dict(size=16)),
        yaxis_title='Harga (USD)', height=400,
        yaxis_tickformat='$,.0f'
    )
    return fig


def create_test_detail_charts(dates_test, y_true, y_pred, model_name, colors):
    """Create test set detail analysis (line + scatter)."""
    dates_pd = pd.to_datetime(dates_test)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f'Prediksi vs Aktual (Test Set)', f'Scatter: Prediksi vs Aktual'
    ], horizontal_spacing=0.1)

    # Line chart
    fig.add_trace(go.Scatter(x=dates_pd, y=y_true, mode='lines', name='Aktual',
                             line=dict(color=colors[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates_pd, y=y_pred, mode='lines', name='Prediksi',
                             line=dict(color=colors[1], width=2, dash='dash')), row=1, col=1)

    # Scatter plot
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Data Point',
                             marker=dict(color=colors[0], size=5, opacity=0.6,
                                         line=dict(color='white', width=0.5)),
                             showlegend=False), row=1, col=2)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Garis Ideal',
                             line=dict(color='#e74c3c', width=2, dash='dash'),
                             showlegend=False), row=1, col=2)

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    fig.add_annotation(text=f'r = {corr:.4f}', xref='x2', yref='y2',
                       x=mn + (mx - mn) * 0.1, y=mx - (mx - mn) * 0.05,
                       showarrow=False, font=dict(size=14, color='#64ffda'),
                       bgcolor='rgba(0,0,0,0.5)', borderpad=6)

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f'Analisis Detail Test Set - {model_name}', font=dict(size=16)),
        height=400
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
    return fig


def metric_card_html(label, value, style_class="metric-value"):
    """Generate metric card HTML."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{style_class}">{value}</div>
    </div>
    """


def forecast_next_30_days(model, data_dict, df):
    """Forecast next 30 days using iterative rolling window."""
    LOOKBACK = 30
    FEATURES = ['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Day_of_Week', 'Month']
    scaler_X = data_dict['scaler_y']  # reuse scaler_y for inverse

    # Build full scaler_X from the same pipeline logic
    from sklearn.preprocessing import MinMaxScaler
    data_feat = df[FEATURES].values
    n_total_raw = len(data_feat)
    TRAIN_RATIO = 0.70
    n_train_raw = int((n_total_raw - LOOKBACK) * TRAIN_RATIO) + LOOKBACK
    sc_X = MinMaxScaler(feature_range=(0, 1))
    sc_X.fit(data_feat[:n_train_raw])
    scaled_full = sc_X.transform(data_feat)

    sc_y = data_dict['scaler_y']

    # Start window: last LOOKBACK rows of scaled data
    window = scaled_full[-LOOKBACK:].copy()  # shape (30, n_features)

    last_date = pd.to_datetime(df['Date'].values[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

    predictions = []
    for _ in range(30):
        X_input = window[np.newaxis, :, :]  # (1, 30, n_features)
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        pred_price = sc_y.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(pred_price)

        # Roll window: shift left and append new row
        # For simplicity, keep non-Close features same as last row, update Close feature (index 0)
        new_row = window[-1].copy()
        new_row[0] = pred_scaled  # update Close feature
        window = np.vstack([window[1:], new_row])

    return future_dates, np.array(predictions)


def create_forecast_chart(future_dates, forecast_prices, last_actual_dates, last_actual_prices, model_name, color_pred):
    """Create 30-day future forecast chart with last 30 actual prices as context."""
    fig = go.Figure()

    # Last actual prices (context)
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(last_actual_dates),
        y=last_actual_prices,
        mode='lines',
        name='Harga Aktual (Terakhir)',
        line=dict(color='#ccd6f6', width=2),
        hovertemplate='%{x|%d %b %Y}<br>Aktual: $%{y:,.2f}<extra></extra>'
    ))

    # Connecting line from last actual to first forecast
    connect_x = [pd.to_datetime(last_actual_dates[-1]), future_dates[0]]
    connect_y = [last_actual_prices[-1], forecast_prices[0]]
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode='lines', showlegend=False,
        line=dict(color=color_pred, width=2, dash='dot')
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_prices,
        mode='lines+markers',
        name=f'Prediksi 30 Hari ({model_name})',
        line=dict(color=color_pred, width=2.5),
        marker=dict(size=5, color=color_pred),
        fill='tonexty',
        fillcolor=f'rgba({int(color_pred[1:3], 16)},{int(color_pred[3:5], 16)},{int(color_pred[5:7], 16)},0.08)',
        hovertemplate='%{x|%d %b %Y}<br>Prediksi: $%{y:,.2f}<extra></extra>'
    ))

    # Vertical separator
    fig.add_vline(
        x=pd.to_datetime(last_actual_dates[-1]).timestamp() * 1000,
        line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dash')
    )
    fig.add_annotation(
        x=future_dates[15], y=max(forecast_prices),
        text="⬅ Aktual  |  Prediksi ➡",
        showarrow=False,
        font=dict(size=12, color='#8892b0'),
        bgcolor='rgba(0,0,0,0.4)', borderpad=6
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f'🔭 Prediksi Harga Bitcoin 30 Hari ke Depan — {model_name}', font=dict(size=16, color='#f5f5f5')),
        yaxis_title='Harga (USD)', height=420,
        yaxis_tickformat='$,.0f'
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="header-title">
        <h1>₿ Dashboard Prediksi Bitcoin</h1>
        <p>Perbandingan Model GRU vs LSTM — Tugas Akhir</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner('⏳ Mengunduh data Bitcoin & memuat model...'):
        df_raw, df = load_and_process_data()
        gru_model, lstm_model = load_models()
        data_dict = prepare_data_for_prediction(df)

        gru_preds = get_predictions(gru_model, data_dict)
        lstm_preds = get_predictions(lstm_model, data_dict)

        gru_metrics_test = calculate_metrics(gru_preds['y_true_test'], gru_preds['y_pred_test'])
        lstm_metrics_test = calculate_metrics(lstm_preds['y_true_test'], lstm_preds['y_pred_test'])

    # ── ROW 1: Bitcoin History ──
    st.markdown('<div class="section-title">📊 Riwayat Harga Bitcoin</div>', unsafe_allow_html=True)
    st.plotly_chart(create_history_chart(df_raw), use_container_width=True)

    st.markdown("---")

    # ── ROW 2: Prediction Charts (GRU then LSTM, full width) ──
    st.markdown('<div class="section-title">🔮 Hasil Prediksi vs Harga Aktual Bitcoin</div>', unsafe_allow_html=True)
    st.plotly_chart(
        create_prediction_chart(
            data_dict['dates_test'],
            gru_preds['y_true_test'], gru_preds['y_pred_test'],
            'GRU', '#3498db', '#f7931a'
        ), use_container_width=True
    )
    st.plotly_chart(
        create_prediction_chart(
            data_dict['dates_test'],
            lstm_preds['y_true_test'], lstm_preds['y_pred_test'],
            'LSTM', '#9b59b6', '#a855f7'
        ), use_container_width=True
    )

    st.markdown("---")

    # ── ROW 3: Test Set Detail Analysis (GRU then LSTM, full width) ──
    st.markdown('<div class="section-title">🔍 Analisis Detail Hasil Test Set</div>', unsafe_allow_html=True)
    st.plotly_chart(
        create_test_detail_charts(
            data_dict['dates_test'],
            gru_preds['y_true_test'], gru_preds['y_pred_test'],
            'GRU', ['#f7931a', '#e74c3c']
        ), use_container_width=True
    )
    st.plotly_chart(
        create_test_detail_charts(
            data_dict['dates_test'],
            lstm_preds['y_true_test'], lstm_preds['y_pred_test'],
            'LSTM', ['#a855f7', '#1abc9c']
        ), use_container_width=True
    )

    st.markdown("---")

    # ── ROW 3b: 30-Day Future Forecast ──
    st.markdown('<div class="section-title">🔭 Prediksi Harga Bitcoin 30 Hari ke Depan</div>', unsafe_allow_html=True)

    with st.spinner('⏳ Menghitung prediksi 30 hari ke depan...'):
        gru_future_dates, gru_forecast = forecast_next_30_days(gru_model, data_dict, df)
        lstm_future_dates, lstm_forecast = forecast_next_30_days(lstm_model, data_dict, df)

    # Context: last 60 actual prices
    last_actual_dates = data_dict['dates_test'][-60:]
    last_actual_prices = gru_preds['y_true_test'][-60:]

    st.plotly_chart(
        create_forecast_chart(gru_future_dates, gru_forecast, last_actual_dates, last_actual_prices, 'GRU', '#f7931a'),
        use_container_width=True
    )
    st.plotly_chart(
        create_forecast_chart(lstm_future_dates, lstm_forecast, last_actual_dates, last_actual_prices, 'LSTM', '#a855f7'),
        use_container_width=True
    )

    st.markdown("---")

    # ── ROW 4: Evaluation Metrics (Test Only) ──
    st.markdown('<div class="section-title">📋 Perbandingan Metrik Evaluasi (Test Set)</div>', unsafe_allow_html=True)

    # --- RMSE & MAE: Grouped Bar Chart (full width) ---
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='GRU',
        x=['RMSE (USD)', 'MAE (USD)'],
        y=[gru_metrics_test['RMSE'], gru_metrics_test['MAE']],
        marker=dict(
            color=['#f7931a', '#ffb347'],
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"${gru_metrics_test['RMSE']:,.2f}", f"${gru_metrics_test['MAE']:,.2f}"],
        textposition='outside',
        textfont=dict(color='#f7931a', size=14, family='Inter'),
        hovertemplate='<b>GRU</b><br>%{x}: $%{y:,.2f}<extra></extra>'
    ))
    fig_bar.add_trace(go.Bar(
        name='LSTM',
        x=['RMSE (USD)', 'MAE (USD)'],
        y=[lstm_metrics_test['RMSE'], lstm_metrics_test['MAE']],
        marker=dict(
            color=['#a855f7', '#c084fc'],
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"${lstm_metrics_test['RMSE']:,.2f}", f"${lstm_metrics_test['MAE']:,.2f}"],
        textposition='outside',
        textfont=dict(color='#a855f7', size=14, family='Inter'),
        hovertemplate='<b>LSTM</b><br>%{x}: $%{y:,.2f}<extra></extra>'
    ))
    fig_bar.update_layout(
        **CHART_LAYOUT,
        title=dict(text='📊 Perbandingan RMSE & MAE — GRU vs LSTM', font=dict(size=16, color='#f5f5f5')),
        barmode='group',
        bargap=0.25,
        bargroupgap=0.1,
        yaxis_title='Nilai Error (USD)',
        yaxis_tickformat='$,.0f',
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- MAPE & Accuracy: Cards (full width, 4 cols) ---
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card_html("🟠 GRU — MAPE (%)", f"{gru_metrics_test['MAPE']:.4f}%", "metric-value metric-value-gru"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("🟠 GRU — Akurasi (%)", f"{gru_metrics_test['Accuracy']:.4f}%", "metric-value metric-value-gru"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("🟣 LSTM — MAPE (%)", f"{lstm_metrics_test['MAPE']:.4f}%", "metric-value metric-value-lstm"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html("🟣 LSTM — Akurasi (%)", f"{lstm_metrics_test['Accuracy']:.4f}%", "metric-value metric-value-lstm"), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#8892b0; font-size:13px; padding:16px 0;">
        Dashboard Tugas Akhir — Prediksi Harga Bitcoin menggunakan GRU & LSTM<br>
        Data: Yahoo Finance (BTC-USD) | Framework: Streamlit + Plotly + TensorFlow
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
