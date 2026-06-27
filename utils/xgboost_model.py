"""
Modul prediksi harga saham menggunakan XGBoost.

Pendekatan:
- Target prediksi adalah log-return harian (bukan harga absolut), karena
  return jauh lebih stasioner dibanding harga sehingga model lebih
  mudah belajar pola yang generalizable antar saham dengan skala harga
  yang berbeda-beda.
- Feature set terdiri dari lag return, rolling mean/std return, rolling
  momentum, RSI, dan rasio harga terhadap moving average.
- Forecast multi-hari ke depan dilakukan secara recursive: prediksi
  hari t+1 dipakai sebagai salah satu input untuk menyusun fitur hari
  t+2, dan seterusnya, sampai horizon yang diminta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

N_LAGS = 10
ROLLING_WINDOWS = (5, 10, 20)
RSI_WINDOW = 14


def _rsi(price: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_feature_frame(price: pd.Series) -> pd.DataFrame:
    """
    Menyusun tabel fitur dari satu deret harga penutupan, TANPA kolom
    target. Baris pertama (sejumlah lag/rolling window terbesar) akan
    mengandung NaN dan dibuang, namun baris TERAKHIR selalu dipertahankan
    apa pun terjadi — ini penting karena baris terakhir adalah basis
    fitur untuk prediksi hari berikutnya saat forecasting.
    """
    df = pd.DataFrame(index=price.index)
    log_price = np.log(price)
    log_return = log_price.diff()

    for lag in range(1, N_LAGS + 1):
        df[f"lag_return_{lag}"] = log_return.shift(lag - 1)

    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = log_return.rolling(w).mean()
        df[f"roll_std_{w}"] = log_return.rolling(w).std()
        df[f"price_to_ma_{w}"] = price / price.rolling(w).mean() - 1

    df["rsi"] = _rsi(price)
    df["momentum_10"] = price / price.shift(10) - 1

    df = df.dropna()
    return df


def build_training_frame(price: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """
    Menyusun tabel fitur LENGKAP DENGAN target untuk keperluan training.
    Target ('target_return') adalah log-return pada hari berikutnya,
    sehingga baris terakhir (yang tidak memiliki hari berikutnya) wajar
    dibuang di sini — namun ini terpisah dari `build_feature_frame` yang
    dipakai saat inferensi, sehingga baris terakhir tetap tersedia untuk
    forecasting.
    """
    feat = build_feature_frame(price)
    feature_cols = list(feat.columns)

    log_return = np.log(price).diff()
    feat = feat.copy()
    feat["target_return"] = log_return.shift(-1).loc[feat.index]
    feat = feat.dropna(subset=["target_return"])
    return feat, feature_cols


def _mape_on_price(
    price: pd.Series, positions: np.ndarray, pred_returns: np.ndarray
) -> float:
    """
    Menghitung MAPE pada skala harga (bukan return) untuk satu himpunan
    posisi (train atau test). `positions` adalah indeks posisi baris T
    pada deret `price`, sedangkan `pred_returns` adalah prediksi
    log-return dari T ke T+1 untuk posisi yang bersesuaian.
    """
    next_positions = positions + 1
    valid_mask = next_positions < len(price)

    base_price = price.iloc[positions[valid_mask]].values
    actual_price_next = price.iloc[next_positions[valid_mask]].values
    pred_price_next = base_price * np.exp(pred_returns[valid_mask])

    return float(
        np.mean(np.abs((actual_price_next - pred_price_next) / actual_price_next)) * 100
    )


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mengukur persentase ketepatan arah pergerakan (naik/turun) antara
    return aktual dan return prediksi — relevan secara praktis karena
    keputusan alokasi portofolio lebih bergantung pada arah pergerakan
    harga dibanding besaran errornya secara presisi.
    """
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(true_dir == pred_dir) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Koefisien determinasi (R^2) pada skala log-return, menunjukkan
    proporsi variansi return yang dapat dijelaskan oleh model relatif
    terhadap baseline rata-rata.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)


@dataclass
class TrainResult:
    model: XGBRegressor
    feature_cols: list[str]
    train_rmse: float
    test_rmse: float
    test_mape: float
    y_test: pd.Series
    y_pred_test: pd.Series
    # --- Metrik evaluasi tambahan (skala return, kecuali disebutkan lain) ---
    train_mae: float
    test_mae: float
    train_mape: float
    train_r2: float
    test_r2: float
    test_directional_accuracy: float


def train_xgboost_model(
    price: pd.Series,
    test_size: float = 0.15,
    n_estimators: int = 150,
    max_depth: int = 4,
    learning_rate: float = 0.08,
) -> TrainResult:
    """
    Melatih model XGBoost untuk satu saham menggunakan train/test split
    kronologis (bukan acak) agar konsisten dengan sifat data time series:
    seluruh data test berada setelah data train secara waktu.
    """
    feat, feature_cols = build_training_frame(price)

    n = len(feat)
    n_test = max(int(n * test_size), 20)
    train_df = feat.iloc[:-n_test]
    test_df = feat.iloc[-n_test:]

    X_train, y_train = train_df[feature_cols], train_df["target_return"]
    X_test, y_test = test_df[feature_cols], test_df["target_return"]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_rmse = float(np.sqrt(np.mean((pred_train - y_train.values) ** 2)))
    test_rmse = float(np.sqrt(np.mean((pred_test - y_test.values) ** 2)))
    train_mae = float(np.mean(np.abs(pred_train - y_train.values)))
    test_mae = float(np.mean(np.abs(pred_test - y_test.values)))
    train_r2 = r_squared(y_train.values, pred_train)
    test_r2 = r_squared(y_test.values, pred_test)
    test_directional_accuracy = directional_accuracy(y_test.values, pred_test)

    # MAPE dihitung pada level harga (bukan return) agar lebih mudah
    # diinterpretasikan. PENTING: target_return pada baris bertanggal T
    # merepresentasikan return dari T ke T+1 (lihat build_training_frame),
    # sehingga harga dasar yang benar untuk merekonstruksi prediksi adalah
    # price[T] (bukan price[T-1]) dan harga aktual yang dibandingkan adalah
    # price[T+1] (bukan price[T]).
    train_positions = price.index.get_indexer(train_df.index)
    test_positions = price.index.get_indexer(test_df.index)

    train_mape = _mape_on_price(price, train_positions, pred_train)
    mape = _mape_on_price(price, test_positions, pred_test)

    return TrainResult(
        model=model,
        feature_cols=feature_cols,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        test_mape=mape,
        y_test=y_test,
        y_pred_test=pd.Series(pred_test, index=y_test.index),
        train_mae=train_mae,
        test_mae=test_mae,
        train_mape=train_mape,
        train_r2=train_r2,
        test_r2=test_r2,
        test_directional_accuracy=test_directional_accuracy,
    )


def forecast_future(
    price: pd.Series,
    model: XGBRegressor,
    feature_cols: list[str],
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Melakukan recursive forecasting sejauh `horizon` hari trading ke
    depan. Pada setiap langkah, fitur dihitung ulang dari deret harga
    yang sudah disambung dengan harga hasil prediksi langkah-langkah
    sebelumnya, kemudian model memprediksi log-return langkah berikutnya.
    """
    history = price.copy()
    last_date = history.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    preds = []
    for d in future_dates:
        feat = build_feature_frame(history)
        x_last = feat[feature_cols].iloc[[-1]]
        next_return = float(model.predict(x_last)[0])
        next_price = history.iloc[-1] * np.exp(next_return)
        history.loc[d] = next_price
        preds.append({"Date": d, "PredictedClose": next_price, "PredictedReturn": next_return})

    return pd.DataFrame(preds).set_index("Date")


def get_expected_returns_from_predictions(
    forecast_df: pd.DataFrame, last_actual_price: float
) -> float:
    """
    Mengonversi hasil forecast horizon-N menjadi satu angka expected
    return kumulatif (dari harga aktual terakhir ke harga prediksi akhir
    horizon). Angka ini yang nantinya menjadi "view" XGBoost pada model
    Black-Litterman.
    """
    final_pred_price = forecast_df["PredictedClose"].iloc[-1]
    return float(final_pred_price / last_actual_price - 1)
