#!/home/reyvn/Dev/RFR_Sugar_Price/.venv/bin/python

import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from hijridate import Gregorian
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf

matplotlib.use("module://matplotlib-backend-kitty")

RAW_DATASET_DIR = Path("../dataset/raw")
DATA_OUTPUT_DIR = Path("../dataset/processed")
PREP_OUTPUT_DIR = DATA_OUTPUT_DIR / "preparation"
PLOT_OUTPUT_DIR = Path("../output/plots")

RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RFR_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)


def print_data(df, title):
    print(title)
    print(df.head())
    print("...")
    print(df.tail())
    print("\n")


def get_dataset() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    paths = sorted(RAW_DATASET_DIR.glob("*.xlsx"))

    if not paths:
        raise FileNotFoundError(
            f"Tidak ditemukan dataset .xlsx pada {RAW_DATASET_DIR.resolve()}"
        )

    for path in paths:
        name = path.stem
        print(f"Loading raw dataset: {path.name}")
        df_raw = pd.read_excel(path)
        datasets[name] = df_raw

    return datasets


def cleanup_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    all_cols = df.columns.tolist()
    num_col = all_cols[0]
    province_col = all_cols[1]

    # Save original row order
    df["prov_order"] = range(len(df))

    # Rename column to province
    df = df.rename(columns={province_col: "Province"})

    # Clean date column names
    date_cols = [c for c in df.columns if c not in [num_col, "Province"]]

    rename_map = {c: c.replace(" ", "") for c in date_cols}
    df = df.rename(columns=rename_map)

    # Recompute date_cols after rename
    date_cols = [c for c in df.columns if c not in [num_col, "Province", "prov_order"]]

    # Melt wide to long format
    df_long = df.melt(
        id_vars=["prov_order", "Province"],
        value_vars=date_cols,
        var_name="Date",
        value_name="Price",
    )

    # Format date to datetime
    df_long["Date"] = pd.to_datetime(df_long["Date"], format="%d/%m/%Y")

    # Replace "-" with NaN & remove comma from all price and convert it to numeric
    df_long["Price"] = (
        df_long["Price"]
        .astype(str)
        .str.strip()
        .replace("-", np.nan)
        .str.replace(",", "", regex=False)
    )
    df_long["Price"] = pd.to_numeric(df_long["Price"], errors="coerce")

    # Sort data
    df_long = df_long.sort_values(["Date", "prov_order"]).reset_index(drop=True)
    df_long = df_long.drop(columns=["prov_order"])
    df_long = df_long[["Date", "Price", "Province"]]

    # Interpolate by province
    df_long["Price"] = df_long.groupby("Province", group_keys=False)["Price"].apply(
        lambda s: s.interpolate(method="linear").ffill().bfill()
    )

    # Fill missing date
    df_long = (
        df_long.groupby("Province", group_keys=False)[
            ["Date", "Price", "Province"]
        ]  # <- perbaikan di sini
        .apply(
            lambda g: (
                g.set_index("Date")
                .reindex(
                    pd.date_range(
                        start=g["Date"].min(),
                        end=g["Date"].max(),
                        freq="D",
                    )
                )
                .assign(Province=lambda x: x["Province"].ffill().bfill())
                .assign(
                    Price=lambda x: x["Price"].interpolate("linear").ffill().bfill()
                )
                .rename_axis("Date")
                .reset_index()
            )
        )
        .reset_index(drop=True)
    )

    # Round price
    df_long["Price"] = df_long["Price"].round().astype(int)

    return df_long


def integrate_datasets(
    cleaned_datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # Pastikan tidak kosong
    if not cleaned_datasets:
        raise ValueError("Datasets kosong, pastikan Data Cleaning sudah dijalankan.")

    frames: list[pd.DataFrame] = []

    for name, df in cleaned_datasets.items():
        temp = df.copy()
        frames.append(temp)

    # Gabung semua dataset
    all_data = pd.concat(frames, ignore_index=True)

    # Normalisasi tipe data
    all_data["Date"] = pd.to_datetime(all_data["Date"])
    all_data["Province"] = all_data["Province"].astype("string")
    all_data["Price"] = pd.to_numeric(all_data["Price"], errors="coerce")

    # Urutkan dan hilangkan duplikat per (Province, Date)
    # kalau ada lebih dari satu sumber di tanggal yg sama, ambil yang terakhir
    all_data = all_data.drop_duplicates(
        subset=["Province", "Date"], keep="last"
    ).reset_index(drop=True)

    print_data(all_data, "Data Integration (Merge All Files) :")

    # Simpan tanpa index
    output_path = PREP_OUTPUT_DIR / f"data_integration.xlsx"
    all_data.to_excel(output_path, index=False)

    return all_data


def plot_sugar_price_history(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ========= OUTPUT PATHS =========
    combined_path = Path("../output/plots/sugar_price_history.png")
    province_dir = Path("../output/plots/province/")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    province_dir.mkdir(parents=True, exist_ok=True)

    # ========= COMBINED PLOT =========
    plt.figure(figsize=(12, 6))

    for prov, g in df.groupby("Province"):
        plt.plot(g["Date"], g["Price"], label=prov, linewidth=0.8, alpha=0.7)

    plt.title("Riwayat Harga Gula per Provinsi")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend(
        title="Provinsi", fontsize=6, ncol=2, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(combined_path)
    plt.close()

    # ========= INDIVIDUAL PROVINCE PLOTS =========
    for prov, g in df.groupby("Province"):
        plt.figure(figsize=(10, 5))
        plt.plot(g["Date"], g["Price"], linewidth=1.5)

        plt.title(f"Riwayat Harga Gula - {prov}")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Rp)")
        plt.tight_layout()

        # clean filename
        file_name = prov.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(province_dir / file_name)
        plt.close()


def compute_significant_pacf_lags(
    series: pd.Series, nlags: int = 60, alpha: float = 0.05
) -> list[int]:
    """
    Hitung lag PACF yang signifikan (koefisien keluar dari interval kepercayaan).
    Dipakai untuk opsi 4: cetak lag signifikan.
    """
    series = series.dropna().astype(float)

    if len(series) <= 1:
        return []

    pacf_vals, confint = pacf(series, nlags=nlags, alpha=alpha, method="ywm")

    sig_lags: list[int] = []
    for lag in range(1, len(pacf_vals)):
        lower, upper = confint[lag]
        val = pacf_vals[lag]
        if val < lower or val > upper:
            sig_lags.append(lag)

    return sig_lags


def plot_pacf_global(df: pd.DataFrame, max_lag: int = 60) -> None:
    """
    PACF untuk harga gula gabungan semua provinsi (median per hari).
    Menyimpan plot + mencetak lag signifikan.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # gabungkan semua provinsi: pakai median harian
    daily_price = df.groupby("Date")["Price"].median().sort_index()

    pacf_dir = PLOT_OUTPUT_DIR / "pacf"
    pacf_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plot_pacf(daily_price, lags=max_lag, method="ywm")
    plt.title("PACF Harga Gula – Gabungan Semua Provinsi")
    plt.tight_layout()

    out_path = pacf_dir / "pacf_global.png"
    plt.savefig(out_path)
    plt.close()

    sig_lags = compute_significant_pacf_lags(daily_price, nlags=max_lag)

    print("\n=== PACF Global (Gabungan Semua Provinsi) ===")
    print(f"File plot : {out_path}")
    print(
        f"Lag signifikan (di luar interval kepercayaan 95%): "
        f"{sig_lags if sig_lags else 'Tidak ada'}\n"
    )


def plot_pacf_per_province(df: pd.DataFrame, max_lag: int = 60) -> None:
    """
    PACF per provinsi.
    Menyimpan plot per-provinsi + mencetak lag signifikan tiap provinsi.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    pacf_dir = PLOT_OUTPUT_DIR / "pacf_province"
    pacf_dir.mkdir(parents=True, exist_ok=True)

    print("=== PACF per Provinsi ===")
    for prov, g in df.groupby("Province"):
        g = g.sort_values("Date")
        series = g["Price"]

        # kalau datanya konstan/pendek banget, skip
        if series.dropna().nunique() <= 1:
            continue

        max_lag_here = min(max_lag, len(series) - 1)
        if max_lag_here <= 0:
            continue

        plt.figure(figsize=(10, 4))
        plot_pacf(series, lags=max_lag_here, method="ywm")
        plt.title(f"PACF Harga Gula – {prov}")
        plt.tight_layout()

        file_name = prov.replace(" ", "_").replace("/", "_") + "_pacf.png"
        out_path = pacf_dir / file_name
        plt.savefig(out_path)
        plt.close()

        sig_lags = compute_significant_pacf_lags(series, nlags=max_lag_here)
        print(
            f"- {prov}: lag signifikan = "
            f"{sig_lags if sig_lags else 'Tidak ada'} "
            f"(plot: {out_path})"
        )
    print()


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pastikan tipe data rapi
    df["Date"] = pd.to_datetime(df["Date"])
    df["Province"] = df["Province"].astype("string")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # 1) Province → numeric (ID)
    df = df.sort_values(["Province", "Date"]).reset_index(drop=True)
    df["Province_id"] = df["Province"].astype("category").cat.codes

    # 2) Lag features per provinsi
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby("Province", group_keys=False)["Price"].shift(lag)

    # 3) Fitur Ramadhan (gunakan hijridate: Hijri month == 9)
    def is_ramadhan(ts: pd.Timestamp) -> int:
        g = Gregorian(ts.year, ts.month, ts.day)
        h = g.to_hijri()
        return 1 if h.month == 9 else 0

    df["ramadhan"] = df["Date"].apply(is_ramadhan).astype(int)

    # Buang baris yang belum punya semua lag (awal-awal tiap provinsi)
    df = df.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_30"]).reset_index(drop=True)

    print_data(df, "Data Transform (Feature Engineering) :")

    output_path = DATA_OUTPUT_DIR / f"data_transformation.xlsx"
    df.to_excel(output_path, index=False)

    return df


def compare_year_prediction(
    target_year: int, df_transformed: pd.DataFrame
) -> pd.DataFrame:
    """
    Train Random Forest using all data BEFORE target_year (all provinces),
    then predict prices for rows in target_year and compare with actual values.

    - Model training: all provinces, all data < target_year
    - Prediction: per row in target_year (all provinces)
    - Output:
        * Excel: ../output/xlsx/rfr_year_comparison_{year}.xlsx
        * Histogram per province:
              ../output/plots/comparison_{year}_{Province}.png
    """
    df = df_transformed.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    feature_cols = ["Province_id", "lag_1", "lag_7", "lag_14", "lag_30", "ramadhan"]
    target_col = "Price"

    # Split train / test by year
    train_mask = df["Date"].dt.year < target_year
    test_mask = df["Date"].dt.year == target_year

    if not train_mask.any():
        raise ValueError(f"Tidak ada data sebelum tahun {target_year} untuk training.")
    if not test_mask.any():
        raise ValueError(f"Tidak ada data pada tahun {target_year} untuk pengujian.")

    df_train = df[train_mask].sort_values("Date")
    df_test = df[test_mask].sort_values("Date")

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # Model dilatih pakai SELURUH PROVINSI (tidak dibatasi provinsi tertentu)
    params = dict(
        n_estimators=600,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluasi keseluruhan tahun
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n=== Year-wise Comparison for {target_year} ===")
    print(f"Train data up to: {df_train['Date'].max().date()}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")

    # Simpan hasil ke Excel
    result_df = pd.DataFrame(
        {
            "Date": df_test["Date"].values,
            "Province": df_test["Province"].values,
            "Actual": y_test.values,
            "Prediction": y_pred,
        }
    )

    out_xlsx_dir = Path("../output/xlsx")
    out_xlsx_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_xlsx_dir / f"rfr_year_comparison_{target_year}.xlsx"
    result_df.to_excel(xlsx_path, index=False)

    # ========== HISTOGRAM PER PROVINSI (COMPARE PRICE, NOT DENSITY) ==========
    hist_dir = Path("../output/plots")

    for prov in sorted(result_df["Province"].unique()):
        sub = result_df[result_df["Province"] == prov]

        if sub.empty:
            continue

        # Sort by date for good line visualization
        sub = sub.sort_values("Date")

        plt.figure(figsize=(12, 5))

        plt.plot(sub["Date"], sub["Actual"], label="Actual", linewidth=1.8)
        plt.plot(sub["Date"], sub["Prediction"], label="Prediction", linewidth=1.8)

        plt.title(f"Riwayat Harga Aktual vs Prediksi - {prov} ({target_year})")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Rp)")
        plt.legend()
        plt.tight_layout()

    print(f"Hasil perbandingan disimpan di: {xlsx_path}")
    print(
        f"Histogram per provinsi disimpan di ../output/plots/comparison_{target_year}_*.png\n"
    )

    return result_df


def run_random_forest_regression(df: pd.DataFrame, train_size) -> RandomForestRegressor:
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by time to keep chronological order for split
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = ["Province_id", "lag_1", "lag_7", "lag_14", "lag_30", "ramadhan"]
    target_col = "Price"

    X = df[feature_cols]
    y = df[target_col]

    # Time-based train-test split
    split_index = int(len(df) * train_size)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(**RFR_PARAMS)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print("\n")
    print("=== Random Forest Regression (Data Mining) ===")
    print(f"Train Data : {train_size*100}%")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")

    # Save prediction results for analysis
    result_df = pd.DataFrame(
        {
            "Date": df.iloc[split_index:]["Date"].values,
            "Province": df.iloc[split_index:]["Province"].values,
            "Actual": y_test.values,
            "Prediction": y_pred,
        }
    )
    Path("../output/xlsx").mkdir(parents=True, exist_ok=True)
    result_df.to_excel("../output/xlsx/rfr_predictions.xlsx", index=False)
    print("Prediction results saved to: ../output/xlsx/rfr_predictions.xlsx")

    return model


def predict_next_year_per_province(
    model: RandomForestRegressor, df_transformed: pd.DataFrame
) -> pd.DataFrame:

    df = df_transformed.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    forecast_results = []

    provinces = df["Province"].unique()

    # == Output folder for plots ==
    forecast_plot_dir = Path("../output/plots/forecast/")
    forecast_plot_dir.mkdir(parents=True, exist_ok=True)

    for prov in provinces:
        g = df[df["Province"] == prov].copy()
        g = g.sort_values("Date").reset_index(drop=True)

        last_date = g["Date"].max()
        last_rows = g.tail(30).copy()

        future_rows = []

        # ======= 365-DAY PREDICTION LOOP =======
        for i in range(1, 365):
            next_date = last_date + pd.Timedelta(days=i)

            lag_1 = last_rows.iloc[-1]["Price"]
            lag_7 = last_rows.iloc[-7]["Price"] if len(last_rows) >= 7 else lag_1
            lag_14 = last_rows.iloc[-14]["Price"] if len(last_rows) >= 14 else lag_1
            lag_30 = last_rows.iloc[-30]["Price"] if len(last_rows) >= 30 else lag_1

            # Ramadhan calculation
            gdate = Gregorian(next_date.year, next_date.month, next_date.day)
            h = gdate.to_hijri()
            ramadhan = 1 if h.month == 9 else 0

            prov_id = g.iloc[-1]["Province_id"]

            X_future = pd.DataFrame(
                {
                    "Province_id": [prov_id],
                    "lag_1": [lag_1],
                    "lag_7": [lag_7],
                    "lag_14": [lag_14],
                    "lag_30": [lag_30],
                    "ramadhan": [ramadhan],
                }
            )

            predicted_price = model.predict(X_future)[0]

            forecast_results.append(
                {
                    "Date": next_date,
                    "Province": prov,
                    "Prediction": round(predicted_price),
                }
            )

            # store for plotting
            future_rows.append({"Date": next_date, "Price": predicted_price})

            # update history for next lag computations
            new_row = {
                "Date": next_date,
                "Price": predicted_price,
                "Province": prov,
                "Province_id": prov_id,
            }
            last_rows = pd.concat(
                [last_rows, pd.DataFrame([new_row])], ignore_index=True
            )

    # ===========================
    # ======= PLOTTING ==========
    # ===========================
    plt.figure(figsize=(12, 5))

    # HISTORICAL DATA
    plt.plot(g["Date"], g["Price"], label="Historical", linewidth=1.5)

    # FUTURE PREDICTION DATA
    future_df = pd.DataFrame(future_rows)
    plt.plot(
        future_df["Date"],
        future_df["Price"],
        label="Forecast (1 Year)",
        linewidth=1.5,
    )

    plt.title(f"Historical + 1-Year Forecast Harga Gula - {prov}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()

    plt.tight_layout()

    file_name = prov.replace(" ", "_").replace("/", "_") + "_forecast.png"
    plot_output_path = PLOT_OUTPUT_DIR / file_name
    plt.savefig(plot_output_path)
    plt.close()

    # ===== Final dataframe output =====
    df_forecast = pd.DataFrame(forecast_results)
    df_forecast = df_forecast.sort_values(["Province", "Date"]).reset_index(drop=True)

    excel_output_path = DATA_OUTPUT_DIR / f"rfr_forecast.xlsx"
    df_forecast.to_excel(excel_output_path, index=False)

    print("\n=== 1-Year Forecast Completed ===")
    print(f"Saved forecast results to {output_path}")
    print(f"Saved forecast charts to ../output/plots/forecast/\n")

    return df_forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--show-history-price",
        action="store_true",
        help="Show and save sugar price history plots",
    )

    parser.add_argument(
        "--plot-pacf",
        action="store_true",
        help="Generate plot PACF global & per provinsi + cetak lag signifikan",
    )

    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Run prediction per province",
    )

    parser.add_argument(
        "--compare-forecast",
        action="store_true",
        help="Run compare prediction",
    )

    args = parser.parse_args()

    # Data Selection
    raw_datasets = get_dataset()

    # Data Cleaning
    cleaned_datasets: dict[str, pd.DataFrame] = {}

    for name, df_raw in raw_datasets.items():
        df_clean = cleanup_dataset(df_raw)
        cleaned_datasets[name] = df_clean

        print_data(df_clean, f"[Data Cleaning] {name}")

        output_path = DATA_OUTPUT_DIR / f"clean_{name}.xlsx"
        df_clean.to_excel(output_path)

    # Data Integration
    df_merged = integrate_datasets(cleaned_datasets)

    if args.show_history_price:
        plot_sugar_price_history(df_merged)

    # PACF (global + per provinsi)
    if args.plot_pacf:
        plot_pacf_global(df_merged)
        plot_pacf_per_province(df_merged)

    # Data Transformation
    df_transformed = transform_dataset(df_merged)

    # Data Mining + Data Evaluation
    run_random_forest_regression(df_transformed, 0.6)
    run_random_forest_regression(df_transformed, 0.7)
    run_random_forest_regression(df_transformed, 0.75)
    run_random_forest_regression(df_transformed, 0.8)
    run_random_forest_regression(df_transformed, 0.9)

    rfr_model = run_random_forest_regression(df_transformed, 0.8)

    # Knowledge Representation
    if args.forecast:
        df_forecast = predict_next_year_per_province(rfr_model, df_transformed)

    if args.compare_forecast:
        compare_year_prediction(2025, df_transformed)
