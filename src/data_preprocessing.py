import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loading import load_data  # Import load_data function

# Load dataset
df = load_data()

def handle_missing_values(data):
    """Fill missing values with column mean."""
    data.fillna(data.mean(), inplace=True)
    print("Missing values handled.")
    return data

def remove_outliers(data):
    """Remove outliers using Interquartile Range (IQR) method."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
    print("Outliers Removed using IQR Method.")
    return data

def normalize_data(data):
    """Normalize data using MinMaxScaler."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)

# Apply preprocessing
df = handle_missing_values(df)
df = remove_outliers(df)
df = normalize_data(df)

# Save the processed data
save_path = "../data/Processed_Heart_Attack_Data.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)
print(f"✅ Processed Dataset Saved at {save_path}")
