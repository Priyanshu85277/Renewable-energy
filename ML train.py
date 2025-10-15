import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation Results:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return mae, rmse, r2

def train_and_evaluate(df, target_col, feature_cols, model_name, model_file):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred, model_name)
    joblib.dump(model, model_file)
    joblib.dump(scaler, model_file.replace(".pkl", "_scaler.pkl"))
    print(f"{model_name} saved as {model_file}")

def main():
    df = pd.read_csv("Dataset/processed_renewable_data.csv")
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}\n")
    df = df.fillna(method='ffill').fillna(method='bfill')
    solar_features = [
        'Temperature', 'Humidity', 'Ground radiation intensity',
        'Radiation intensity in the upper atmosphere'
    ]
    train_and_evaluate(df, 'Photovoltaic power generation', solar_features,
                       "Solar Power Model", "solar_power_model.pkl")
    wind_features = [
        'Air density', 'Wind Speed'
    ]
    train_and_evaluate(df, 'Power generation', wind_features,
                       "Wind Power Model", "wind_power_model.pkl")
    print("\nPhase 3 completed successfully — both models trained and saved.")

if __name__ == "__main__":
    main()
