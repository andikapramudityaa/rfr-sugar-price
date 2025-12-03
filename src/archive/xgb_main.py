#!/home/reyvn/Dev/RFR_Sugar_Price/.venv/bin/python

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from hijridate import Hijri, Gregorian

matplotlib.use("module://matplotlib-backend-kitty")


def get_dataset():
    file_path = "../dataset/sugar_prices.xlsx"
    return pd.read_excel(file_path)


def cleanup_dataset(df):
    print(df.head())
    # Drop unnecessary columns and index then tranpose the data frame
    df = df.drop(columns=["No", "Komoditas (Rp)"]).iloc[[0]].T
    # Reset index
    df.reset_index(inplace=True)
    # Rename header
    df.columns = ["Date", "Price"]

    # Convert column date to datetime format
    df["Date"] = pd.to_datetime(
        df["Date"].str.replace(" ", "").str.strip(),
        format="%d/%m/%Y",
        errors="coerce",
    )

    # Replace "-" with NaN & remove comma from all price and convert it to numeric
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("-", "")
        .str.replace(",", "")
        .replace("", np.nan)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Sort by date just in case
    df = df.sort_values("Date")
    # Generate a full date range from min to max date
    full_dates = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
    # Reindex dataframe to ensure all dates are present
    df = df.set_index("Date").reindex(full_dates).rename_axis("Date").reset_index()
    # Interpolate missing prices (based on nearest valid values)
    df["Price"] = df["Price"].interpolate(method="linear")
    df["Price"] = np.ceil(df["Price"])
    # Forward/Backward fill for any remaining NaN at start/end
    df["Price"] = df["Price"].ffill().bfill()

    print_data(df, "Data Cleaning Result :")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Price"])
    plt.title("Riwayat Harga Gula")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp/kg)")
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("../output/images/sugar_price_history.png", dpi=300)
    plt.show()

    return df


def feature_engineering(data):
    # Time features
    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["weekofyear"] = data["Date"].dt.isocalendar().week.astype(int)
    # Trend (sequential index)
    data["trend"] = np.arange(len(data))
    # Lag features
    for lag in [1, 7, 30, 60, 90]:
        data[f"lag_{lag}_price"] = data["Price"].shift(lag)
    # Delta features
    for delta in [30]:
        data[f"delta_{delta}"] = data["Price"] - data["Price"].shift(delta)
    for pct_delta in [30]:
        data[f"pct_delta_{delta}"] = data["Price"].pct_change(delta)
    # Slope features
    for slope in [7, 30]:
        data[f"slope_{slope}"] = data["Price"].diff(slope)
    # Drop NaN from lag features
    data = data.dropna().reset_index(drop=True)
    # Ramadan feature
    data[["is_ramadan", "ramadan_day", "days_to_eid", "eid_window14"]] = data[
        "Date"
    ].apply(_ramadan_eid_feats)

    # Print feature engineering result
    print("Columns created :", list(data.columns))
    print_data(data, "Feature engineering result :")

    return data


def build_model(data, use_full_data, test_ratio=0.2):
    feature_cols = [
        col
        for col in data.columns
        if col not in ["Date", "Price"] and np.issubdtype(data[col].dtype, np.number)
    ]
    target_col = "Price"

    X = data[feature_cols]
    y = data[target_col]

    if use_full_data:
        # pakai semua data untuk train (untuk forecasting / compare_prediction)
        X_train, y_train = X, y
        X_test = y_test = None
    else:
        # mode evaluasi internal (kalau kamu mau cek performa umum)
        split_index = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training samples: {len(X_train)}")

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=None,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    # evaluasi internal kalau X_test tidak None
    if X_test is not None and len(X_test) > 0:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        print("\nModel Evaluation Results")
        print(f"RMSE : {rmse:,.2f}")
        print(f"MAPE : {mape:.2f}%")

    return model, feature_cols


def _ramadan_eid_feats(date_like):
    d = pd.Timestamp(date_like)
    h = Gregorian(d.year, d.month, d.day).to_hijri()

    # 1) Ramadan flag and day-of-Ramadan (1..30)
    is_ramadan = 1 if int(h.month) == 9 else 0
    ramadan_day = int(h.day) if is_ramadan else 0

    # 2) Days to Eid al-Fitr (1 Shawwal) using same Hijri year; if already far past, use next year
    eid_same = Hijri(int(h.year), 10, 1).to_gregorian()  # 1 Shawwal
    eid_same_ts = pd.Timestamp(
        int(eid_same.year), int(eid_same.month), int(eid_same.day)
    )
    diff = (eid_same_ts - d).days
    if diff < -15:
        eid_next = Hijri(int(h.year) + 1, 10, 1).to_gregorian()
        eid_ts = pd.Timestamp(
            int(eid_next.year), int(eid_next.month), int(eid_next.day)
        )
        diff = (eid_ts - d).days
    else:
        eid_ts = eid_same_ts

    # 3) 14-day moving-holiday window around Eid: 7 days before through 6 days after
    eid_window14 = 1 if -6 <= diff <= 7 else 0

    return pd.Series(
        {
            "is_ramadan": int(is_ramadan),
            "ramadan_day": int(ramadan_day),
            "days_to_eid": int(diff),
            "eid_window14": int(eid_window14),
        }
    )


def forecast_future_prices(model, data, horizon_days, feature_cols):
    df_future = data.copy()
    for _ in range(horizon_days):
        next_date = df_future["Date"].iloc[-1] + pd.Timedelta(days=1)

        # copy last row
        new_row = df_future.iloc[-1].copy()
        new_row["Date"] = next_date

        # update delta features
        for d in [30]:
            d_col = f"delta_{d}"
            if d_col in feature_cols:
                new_row[d_col] = new_row["Price"] - (
                    df_future["Price"].iloc[-d]
                    if len(df_future) > d
                    else df_future["Price"].iloc[0]
                )

        # update percentage deltas (optional)
        for d in [30]:
            p_col = f"pct_delta_{d}"
            if p_col in feature_cols:
                past = (
                    df_future["Price"].iloc[-d]
                    if len(df_future) > d
                    else df_future["Price"].iloc[0]
                )
                new_row[p_col] = 0 if past == 0 else (new_row["Price"] - past) / past

        for d in [7, 30]:
            d_col = f"slope_{d}"
            if d_col in feature_cols:
                past = (
                    df_future["Price"].iloc[-d]
                    if len(df_future) > d
                    else df_future["Price"].iloc[0]
                )
                new_row[d_col] = new_row["Price"] - past

        # update time features
        new_row["year"] = next_date.year
        new_row["month"] = next_date.month
        new_row["weekofyear"] = int(pd.Timestamp(next_date).isocalendar().week)
        new_row["trend"] = int((next_date - data["Date"].min()).days)

        h = Gregorian(next_date.year, next_date.month, next_date.day).to_hijri()
        new_row["is_ramadan"] = 1 if int(h.month) == 9 else 0
        new_row["ramadan_day"] = int(h.day) if new_row["is_ramadan"] == 1 else 0

        eid_same = Hijri(int(h.year), 10, 1).to_gregorian()
        eid_same_ts = pd.Timestamp(
            int(eid_same.year), int(eid_same.month), int(eid_same.day)
        )
        _days = (eid_same_ts - next_date).days
        if _days < -15:
            eid_next = Hijri(int(h.year) + 1, 10, 1).to_gregorian()
            eid_ts = pd.Timestamp(
                int(eid_next.year), int(eid_next.month), int(eid_next.day)
            )
            _days = (eid_ts - next_date).days
        new_row["days_to_eid"] = int(_days)
        new_row["eid_window14"] = 1 if -6 <= _days <= 7 else 0

        # predict
        X_next = pd.DataFrame([new_row[feature_cols].values], columns=feature_cols)
        y_next = model.predict(X_next)[0]
        new_row["Price"] = y_next

        # update lag features using latest predicted prices
        for lag in [1, 7, 30]:
            lag_col = f"lag_{lag}_price"
            if lag_col in feature_cols:
                new_row[lag_col] = (
                    df_future["Price"].iloc[-lag]
                    if len(df_future) > lag
                    else df_future["Price"].iloc[0]
                )

        # append predicted row to future data
        df_future = pd.concat([df_future, pd.DataFrame([new_row])], ignore_index=True)

    return df_future.tail(horizon_days)[["Date", "Price"]].rename(
        columns={"Price": "PredictedPrice"}
    )


def do_forecast(model, data, feature_cols):
    future_range = 365 * 3
    future_prices = forecast_future_prices(model, data, future_range, feature_cols)

    print_data(future_prices, "Future Prediction : ")

    future_prices.to_csv("../output/csv/future_prices.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Price"], label="History")
    plt.plot(future_prices["Date"], future_prices["PredictedPrice"], label="Forecast")
    plt.title("Sugar Price - History & Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (Rp/kg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../output/images/xgb_forecast_future_prices.png", dpi=300)
    plt.show()


# Compare prediction with actual data for a target year
def compare_prediction(data_full, target_year):
    start_of_year = pd.Timestamp(year=target_year, month=1, day=1)
    end_of_year = pd.Timestamp(year=target_year, month=12, day=31)

    base_data = data_full[data_full["Date"] < start_of_year].copy()
    actual_year = data_full[
        (data_full["Date"] >= start_of_year) & (data_full["Date"] <= end_of_year)
    ][["Date", "Price"]].copy()

    if base_data.empty:
        print(
            f"[Error] No historical data before {target_year} to build the forecast base."
        )
        return None
    if actual_year.empty:
        print(f"[Error] No actual data found for year {target_year} in the dataset.")
        return None

    # Train model
    model_year, feature_cols_year = build_model(base_data, True)

    # Forecast
    horizon_days = len(actual_year)
    preds_year = forecast_future_prices(
        model_year, base_data, horizon_days, feature_cols_year
    )

    # Align dates
    comp = (
        actual_year.set_index("Date")
        .join(preds_year.set_index("Date"), how="inner")
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    # Clean NaNs and ensure we have something to score
    comp = comp.dropna(subset=["Price", "PredictedPrice"])
    if comp.empty:
        print(
            "[Error] No overlapping dates between actual and predicted after alignment."
        )
        return None

    # Metric
    rmse = np.sqrt(mean_squared_error(comp["Price"], comp["PredictedPrice"]))
    mape = mean_absolute_percentage_error(comp["Price"], comp["PredictedPrice"]) * 100

    print(f"\n=== Comparison for Year {target_year} ===")
    print(f"Overlapping days: {len(comp):,}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(comp.head())

    # Visualization
    out_csv = f"../output/csv/compare_actual_vs_predicted_{target_year}.csv"
    comp.to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(comp["Date"], comp["Price"], label=f"Actual Price ({target_year})")
    plt.plot(
        comp["Date"],
        comp["PredictedPrice"],
        label=f"Predicted Price ({target_year})",
        linestyle="--",
    )
    plt.title(f"Sugar Price: Prediction vs Actual ({target_year})")
    plt.xlabel("Date")
    plt.ylabel("Price (Rp/kg)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        f"../output/images/xgboost_compare_actual_vs_predicted_{target_year}.png",
        dpi=300,
    )
    plt.show()

    return comp


def print_data(df, title):
    print(title)
    print(df.head())
    print("...")
    print(df.tail())


df = get_dataset()
df = cleanup_dataset(df)
data = feature_engineering(df)
model, features = build_model(data, False, 0.1)
model, features = build_model(data, False, 0.2)
model, features = build_model(data, False, 0.25)
model, features = build_model(data, False, 0.3)
model, features = build_model(data, False, 0.4)
# do_forecast(model, data, features)
compare_prediction(data, target_year=2025)
