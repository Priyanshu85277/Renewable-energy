import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_plot(df, features, target_col, model_path, scaler_path, title_prefix):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X = df[features]
    y_true = df[target_col]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{title_prefix} Model Performance:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_true.values[:200], label='Actual', color='blue')
    plt.plot(y_pred[:200], label='Predicted', color='red', linestyle='dashed')
    plt.title(f"{title_prefix} Power Generation (Sample 200 Points)")
    plt.xlabel("Time Steps")
    plt.ylabel("Power Generation")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, color='green')
    plt.title(f"{title_prefix}: Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("Dataset/processed_renewable_data.csv")
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    df = df.ffill().bfill()
    solar_features = [
        'Temperature', 'Humidity',
        'Ground radiation intensity',
        'Radiation intensity in the upper atmosphere'
    ]
    evaluate_and_plot(
        df,
        features=solar_features,
        target_col='Photovoltaic power generation',
        model_path='solar_power_model.pkl',
        scaler_path='solar_power_model_scaler.pkl',
        title_prefix='Solar'
    )
    wind_features = [
        'Air density', 'Wind Speed'
    ]
    evaluate_and_plot(
        df,
        features=wind_features,
        target_col='Power generation',
        model_path='wind_power_model.pkl',
        scaler_path='wind_power_model_scaler.pkl',
        title_prefix='Wind'
    )

if __name__ == "__main__":
    main()
