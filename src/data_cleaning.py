# src/data_cleaning.py

import numpy as np
import pandas as pd

KNOWN_MAKES = [
    "MERCEDES-BENZ",
    "LAND ROVER",
    "ALFA ROMEO",
    "ASTON MARTIN",
    "ROLLS-ROYCE",
    "VOLKSWAGEN",
    "MITSUBISHI",
    "CHEVROLET",
    "LAMBORGHINI",
    "MASERATI",
    "DAIHATSU",
    "TOYOTA",
    "SUBARU",
    "SUZUKI",
    "NISSAN",
    "PORSCHE",
    "PEUGEOT",
    "RENAULT",
    "TESLA",
    "LEXUS",
    "HONDA",
    "VOLVO",
    "MAZDA",
    "JEEP",
    "BMW",
    "AUDI",
    "MINI",
    "FIAT",
    "ISUZU",
]

# Helper functions


# 1. title cleaning
def extract_make(title: str) -> str:
    title = str(title).upper()

    for make in KNOWN_MAKES:
        if make in title:
            return make
    return "OTHER"


# 2. mileage cleaning
def clean_mileage(x):
    return pd.to_numeric(str(x).replace(",", "").replace("km", ""), errors="coerce")


# 3. engine
def clean_engine(x):
    return pd.to_numeric(str(x).replace(",", "").replace("cc", ""), errors="coerce")


# 4. price
def clean_price(x):
    return pd.to_numeric(str(x).replace("US$", "").replace(",", ""), errors="coerce")


# Main pipeline


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic parsing
    parts = df["title"].str.split(" ", n=2, expand=True)
    df["make"] = df["title"].apply(extract_make)

    df["model"] = parts[2]

    # Numeric cleaning
    df["mileage_km"] = df["mileage"].apply(clean_mileage)
    df["engine_cc"] = df["engine_cc"].apply(clean_engine)
    df["price_usd"] = df["price_usd"].apply(clean_price)

    df["year"] = pd.to_numeric(
        df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )

    # Derived features
    df["car_age"] = 2026 - df["year"]
    df["shipping_cost"] = df["total_price_usd"] - df["price_usd"]

    # Transmission mapping
    df["transmission"] = df["transmission"].astype(str).str.strip()

    df["transmission_type"] = df["transmission"].replace(
        {
            "AT": "Automatic",
            "CVT": "Automatic",
            "Semi AT": "Semi-Automatic",
            "MT": "Manual",
        }
    )

    # Missing / cleanup rules
    df["model_code"] = df["model_code"].replace(["-", ""], np.nan)

    df["engine_code"] = df["engine_code"].replace("-", np.nan)

    df["seats"] = pd.to_numeric(
        df["seats"].replace(["ASK", "-", "99", "5228", ""], np.nan), errors="coerce"
    )

    df["doors"] = pd.to_numeric(df["doors"], errors="coerce")

    df["destination_port"] = df["destination_port"].fillna("Unknown")

    # Accessories cleanup
    df["accessories"] = df["accessories"].fillna("")

    # IDs cleanup
    df["ref_no"] = df["ref_no"].str.replace("Ref No. ", "", regex=False)

    # Final column order
    columns = [
        "make",
        "model",
        "model_code",
        "year",
        "car_age",
        "engine_cc",
        "engine_code",
        "fuel",
        "transmission_type",
        "drive",
        "mileage_km",
        "steering",
        "color",
        "seats",
        "doors",
        "price_usd",
        "total_price_usd",
        "shipping_cost",
        "location",
        "destination_port",
        "accessories",
        "vehicle_id",
        "ref_no",
        "vehicle_url",
    ]

    df = df[columns]

    return df


# CLI
if __name__ == "__main__":
    raw_path = "../data/raw/beforward/raw_beforward_fast_scraped.csv"
    clean_path = "../data/clean/cli_beforward_clean.csv"

    print(r"Starting data cleaning...")
    processed_df = load_and_clean(raw_path)
    processed_df.to_csv(clean_path, index=False)
    print(f"Cleaning complete : ({processed_df.shape})")
