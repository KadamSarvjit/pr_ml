
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# 1. Load dataset
df = pd.read_csv("uber.csv")

print("Original Dataset:")
print(df.head())
print("Dataset Shape:", df.shape)

# 2. Preprocess data
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.drop_duplicates()

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

df = df.dropna(subset=[
    "fare_amount", "pickup_datetime",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count"
])

df = df[
    (df["fare_amount"] > 0) &
    (df["pickup_longitude"].between(-75, -72)) &
    (df["dropoff_longitude"].between(-75, -72)) &
    (df["pickup_latitude"].between(39, 42)) &
    (df["dropoff_latitude"].between(39, 42)) &
    (df["passenger_count"].between(1, 6))
].copy()

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.dayofweek

# Calculate distance using Haversine formula
lat1 = np.radians(df["pickup_latitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lon2 = np.radians(df["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

df["distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
)

# Remove zero-distance rides
df = df[df["distance_km"] > 0].copy()

# 3. Identify fare outliers using IQR
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

lower = max(0, Q1 - 1.5 * IQR)
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["fare_amount"] < lower) |
    (df["fare_amount"] > upper)
]

print("\nNumber of potential fare outliers:", len(outliers))

# Remove potential fare outliers
df = df[df["fare_amount"].between(lower, upper)].copy()

print("Cleaned Dataset Shape:", df.shape)

# 4. Check correlation
features = [
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "hour", "day", "month",
    "year", "weekday", "distance_km"
]

correlation = df[features].corr()

print("\nCorrelation with fare amount:")
print(correlation["fare_amount"].sort_values(ascending=False))

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True)
plt.title("Correlation Heatmap of Uber Fares")
plt.tight_layout()
plt.show()

# 5. Prepare input and target
X = df.drop(columns=["fare_amount", "pickup_datetime"])
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train regression models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(), Ridge(alpha=1.0)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(), Lasso(alpha=0.01, max_iter=10000)
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    results.append([name, r2, rmse, mae])

# 7. Compare model performance
results_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "RMSE", "MAE"]
)

print("\nModel Performance Comparison:")
print(results_df.round(4).to_string(index=False))

# 8. Predict a sample fare using Linear Regression
sample_prediction = models["Linear Regression"].predict(
    X_test.iloc[[0]]
)

print("\nActual Fare:", round(y_test.iloc[0], 2))
print("Predicted Fare:", round(sample_prediction[0], 2))

print("\nUber Fare Prediction Completed Successfully!")
