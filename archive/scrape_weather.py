import pandas as pd
import requests
import time
import os

# Exact coordinates (Latitude, Longitude) for the 34 Indonesian Provincial Capitals
provinces_coords = {
    "Aceh": (5.548278, 95.323753),
    "Sumatera Utara": (3.595196, 98.672226),
    "Sumatera Barat": (-0.947083, 100.369206),
    "Riau": (0.507068, 101.445075),
    "Jambi": (-1.610123, 103.613120),
    "Sumatera Selatan": (-2.990934, 104.756554),
    "Bengkulu": (-3.792845, 102.260764),
    "Lampung": (-5.429394, 105.266296),
    "Kepulauan Bangka Belitung": (-2.120610, 106.113524),
    "Kepulauan Riau": (0.917056, 104.446549),
    "DKI Jakarta": (-6.208763, 106.845599),
    "Jawa Barat": (-6.914744, 107.609810),
    "Jawa Tengah": (-6.981845, 110.416928),
    "DI Yogyakarta": (-7.795580, 110.369490),
    "Jawa Timur": (-7.250445, 112.768845),
    "Banten": (-6.115295, 106.159334),
    "Bali": (-8.670458, 115.212629),
    "Nusa Tenggara Barat": (-8.583333, 116.116667),
    "Nusa Tenggara Timur": (-10.170591, 123.606995),
    "Kalimantan Barat": (-0.027582, 109.333256),
    "Kalimantan Tengah": (-2.216298, 113.905652),
    "Kalimantan Selatan": (-3.316694, 114.590111),
    "Kalimantan Timur": (-0.502182, 117.153629),
    "Kalimantan Utara": (2.841527, 117.364560),
    "Sulawesi Utara": (1.489725, 124.842777),
    "Sulawesi Tengah": (-0.891667, 119.870719),
    "Sulawesi Selatan": (-5.147575, 119.432657),
    "Sulawesi Tenggara": (-3.974868, 122.585721),
    "Gorontalo": (0.540194, 123.056345),
    "Sulawesi Barat": (-2.677464, 118.887201),
    "Maluku": (-3.695379, 128.181410),
    "Maluku Utara": (0.730374, 127.585501),
    "Papua Barat": (-0.863386, 134.062024),
    "Papua": (-2.533710, 140.718624),
}

start_date = "2022-01-01"
end_date = "2026-03-01"
output_file = "../output/indonesian_weather_2022_2026.csv"

weather_records = []
completed_provinces = []

# Load existing data so we don't overwrite it or start over
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file)
        # Check if the CSV actually has data and the 'Province' column exists
        if not existing_df.empty and "Province" in existing_df.columns:
            completed_provinces = existing_df["Province"].unique().tolist()
            print(
                f"Found existing data for {len(completed_provinces)} provinces. Resuming..."
            )
            weather_records.append(existing_df)
    except Exception as e:
        print(f"Could not read existing CSV. Starting fresh. Error: {e}")

# Filter out the provinces we already have
remaining_provinces = {
    prov: coords
    for prov, coords in provinces_coords.items()
    if prov not in completed_provinces
}

if not remaining_provinces:
    print("All provinces are already downloaded! The dataset is complete.")
else:
    print(f"Fetching data for the remaining {len(remaining_provinces)} provinces...")

    for prov, (lat, lon) in remaining_provinces.items():
        print(f"Downloading data for {prov}...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,precipitation_sum&timezone=Asia%2FJakarta"
        )

        try:
            # Added a 15-second timeout to prevent the script from freezing
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                daily = data["daily"]

                df_prov = pd.DataFrame(
                    {
                        "Date": pd.to_datetime(daily["time"]),
                        "Province": prov,
                        "Temp_Mean": daily["temperature_2m_mean"],
                        "Precipitation_mm": daily["precipitation_sum"],
                    }
                )
                weather_records.append(df_prov)

                # Save progress iteratively just in case it crashes again
                pd.concat(weather_records, ignore_index=True).to_csv(
                    output_file, index=False
                )

            else:
                print(f"Failed to fetch {prov} - Status Code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Connection error for {prov}: {e}")

        # Increased sleep slightly to keep the API servers happy
        time.sleep(3)

    print(f"\nFinished! Your updated dataset is saved at '{output_file}'.")
