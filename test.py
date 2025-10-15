import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys


sys.stdout.reconfigure(encoding='utf-8')

def test_and_validate(df, features, target_col, model_path, scaler_path, title_prefix):
    """Evaluate a trained model on unseen data."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n>>> {title_prefix} Model — Test Results <<<")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    results.to_csv(f"{title_prefix.lower()}_test_results.csv", index=False)
    print(f"Results saved → {title_prefix.lower()}_test_results.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:200], label='Actual', color='blue')
    plt.plot(y_pred[:200], label='Predicted', color='red', linestyle='--')
    plt.title(f"{title_prefix} Power Generation — Test Comparison (200 Samples)")
    plt.xlabel("Samples")
    plt.ylabel("Power Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading processed dataset...")
    df = pd.read_csv("Dataset/processed_renewable_data.csv")
    print(f"Dataset shape: {df.shape}")

    df = df.ffill().bfill()

    solar_features = [
        'Temperature', 'Humidity',
        'Ground radiation intensity',
        'Radiation intensity in the upper atmosphere'
    ]
    test_and_validate(
        df,
        features=solar_features,
        target_col='Photovoltaic power generation',
        model_path='solar_power_model.pkl',
        scaler_path='solar_power_model_scaler.pkl',
        title_prefix='Solar'
    )

    wind_features = ['Air density', 'Wind Speed']
    test_and_validate(
        df,
        features=wind_features,
        target_col='Power generation',
        model_path='wind_power_model.pkl',
        scaler_path='wind_power_model_scaler.pkl',
        title_prefix='Wind'
    )

    print("\nPhase 6 Completed Successfully — Testing & Validation Done!")


if __name__ == "__main__":
    main()
