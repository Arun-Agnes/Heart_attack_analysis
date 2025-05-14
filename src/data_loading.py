import pandas as pd

def load_data(file_path="../data/Heart_Attack_Analysis_Data.csv"):
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
    print("First 5 Rows:\n", data.head(), "\n")
    print("Shape of Dataset:", data.shape, "\n")
    print("Dataset Description:\n", data.describe(), "\n")
    return data

df = load_data()