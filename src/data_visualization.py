import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loading import load_data  # Import the function from data_loading.py

df = load_data()  # Load the dataset before visualization

def visualize_data(data):
    sns.set_style("whitegrid")
    data.hist(figsize=(12, 8), bins=20, color="blue", edgecolor="black")
    plt.suptitle("Histogram of Features", fontsize=14)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x=data["Target"], hue=data["Target"], palette="coolwarm", legend=False)
    plt.title("Distribution of Target Variable")
    plt.xlabel("Target (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.show()

    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

visualize_data(df)
