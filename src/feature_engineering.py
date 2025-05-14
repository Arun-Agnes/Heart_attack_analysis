import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the processed dataset instead of reapplying preprocessing
df = pd.read_csv("../data/Processed_Heart_Attack_Data.csv")

def feature_importance(data):
    """Compute feature importance using Decision Tree."""
    X = data.drop(columns=['Target'])  # Features
    y = data['Target']  # Target variable
    model = DecisionTreeClassifier()
    model.fit(X, y)
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    print("Feature Importance:\n", feature_importance_df)
    return feature_importance_df

# Perform feature importance analysis
feature_importance(df)
