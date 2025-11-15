#!/home/reyvn/Dev/RFR_Sugar_Price/.venv/bin/python

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re

matplotlib.use("module://matplotlib-backend-kitty")

# ======================
# 1. BACA DATA
# ======================
FILE_PATH = "../dataset/all_commodity_23-24.xlsx"

df_raw = pd.read_excel(FILE_PATH)


# ======================
# 2. TANDAI KATEGORI UMUM vs VARIAN
#    - No = I, II, III, ...  => kategori umum
#    - No = 1, 2, 3, ...    => varian (dibuang)
# ======================
def is_general_row(no_value):
    """
    Mengembalikan True jika kolom 'No' BUKAN angka murni
    (jadi diasumsikan sebagai I, II, III, IV, dst).
    """
    s = str(no_value).strip()
    # fullmatch angka (1, 2, 10, dst) -> varian
    return not bool(re.fullmatch(r"\d+", s))


df_raw["is_general"] = df_raw["No"].apply(is_general_row)

df_general_rows = df_raw[df_raw["is_general"]].copy()

# ======================
# 3. UBAH WIDE → LONG
# ======================
id_cols = ["No", "Komoditas (Rp)"]
value_cols = [
    c
    for c in df_general_rows.columns
    if c not in ["No", "Komoditas (Rp)", "is_general"]
]

df_long = df_general_rows.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="DateStr",
    value_name="PriceRaw",
)

# ======================
# 4. BERSIHKAN TANGGAL & HARGA
# ======================
df_long["DateStr"] = df_long["DateStr"].str.replace(" ", "", regex=False)
df_long["Date"] = pd.to_datetime(df_long["DateStr"], format="%d/%m/%Y", errors="coerce")


def parse_price(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if s in ["", "-"]:
        return pd.NA

    # buang pemisah ribuan lokal
    s = s.replace(".", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return pd.NA


df_long["Price"] = df_long["PriceRaw"].map(parse_price)

# buang baris tanpa tanggal / harga
df_long = df_long.dropna(subset=["Date", "Price"])

# ======================
# 5. FILTER PERIODE 2023–2024
# ======================
mask_period = (df_long["Date"] >= "2023-01-01") & (df_long["Date"] <= "2024-12-31")
df_period = df_long.loc[mask_period].copy()

# ======================
# 6. CEK DAFTAR KOMODITAS YANG AKAN DIPLOT
# ======================
print(
    "Komoditas umum yang dipakai (harusnya termasuk Beras, Daging Ayam, Gula Pasir, dll):"
)
for name in sorted(df_period["Komoditas (Rp)"].unique()):
    print("-", name)

# ======================
# 7. PLOT SEMUA KOMODITAS UMUM
# ======================
plt.figure(figsize=(14, 6))

for name, grp in df_period.groupby("Komoditas (Rp)"):
    g = grp.sort_values("Date")
    plt.plot(g["Date"], g["Price"], label=name)

plt.title("Harga Komoditas Pangan Umum (2023–2024)")
plt.xlabel("Tanggal")
plt.ylabel("Harga (Rp)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
