import os
import pandas as pd
from glob import glob

def load_and_preprocess(folder_path, keyword):
    """Load all CSVs containing the keyword and combine them."""
    csv_files = [f for f in glob(os.path.join(folder_path, "*.csv")) if keyword.lower() in f.lower()]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for keyword '{keyword}' in {folder_path}")
    
    dfs = []
    for file in csv_files:
        print(f"Loading file: {os.path.basename(file)}")
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.sort_values(by=time_col)
        else:
            df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            time_col = 'Time'

        df = df.dropna(subset=[time_col])
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(by=time_col).reset_index(drop=True)
    return combined, time_col


def preprocess_renewable_data(folder_path):
    """Combine, clean, and save renewable datasets."""
    solar_data, solar_time = load_and_preprocess(folder_path, "photovoltaic")
    print(f"Solar dataset processed with shape: {solar_data.shape}")
    
    wind_data, wind_time = load_and_preprocess(folder_path, "wind")
    print(f"Wind dataset processed with shape: {wind_data.shape}")
    
    merged = pd.merge_asof(
        solar_data.sort_values(by=solar_time),
        wind_data.sort_values(by=wind_time),
        left_on=solar_time,
        right_on=wind_time,
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )

    print(f"Merged dataset shape: {merged.shape}")

    merged = merged.ffill().bfill()

    output_path = os.path.join(folder_path, "processed_renewable_data.csv")
    merged.to_csv(output_path, index=False, encoding='utf-8')

    print("Processed dataset successfully saved as 'processed_renewable_data.csv' (UTF-8 encoded).")


def main():
    folder_path = "Dataset"
    preprocess_renewable_data(folder_path)


if __name__ == "__main__":
    main()
