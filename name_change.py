import pandas as pd

# Path to your IMAX list
INPUT_CSV  = "list_of_IMAX.csv"
OUTPUT_CSV = "list_of_IMAX_cleaned.csv"

# Mapping of abbreviations to full country names
COUNTRY_MAP = {
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates"
}

def normalize_country_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Country" not in df.columns:
        raise ValueError("CSV missing 'Country' column")

    # Strip spaces, map known codes, leave others unchanged
    df["Country"] = df["Country"].astype(str).str.strip().apply(
        lambda c: COUNTRY_MAP.get(c, c)
    )
    return df

def main():
    df = pd.read_csv(INPUT_CSV)
    df = normalize_country_names(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Cleaned CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
