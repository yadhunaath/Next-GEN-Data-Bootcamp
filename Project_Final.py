import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("SeoulBikeData.csv", encoding="latin1")

print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.drop_duplicates()
df = df.dropna(subset=["Date"])

df.rename(columns={
    "Temperature(°C)": "Temperature",
    "Humidity(%)": "Humidity",
    "Wind speed (m/s)": "WindSpeed",
    "Visibility (10m)": "Visibility",
    "Dew point temperature(°C)": "DewPoint",
    "Solar Radiation (MJ/m2)": "SolarRadiation",
    "Rainfall(mm)": "Rainfall",
    "Snowfall (cm)": "Snowfall",
    "Functioning Day": "FunctioningDay",
    "Rented Bike Count": "BikeCount"
}, inplace=True)

df["Month"] = df["Date"].dt.month_name()
df["DayOfWeek"] = df["Date"].dt.day_name()
df["Weekend"] = df["Date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
df["PeakHour"] = df["Hour"].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 20) else 0)

plt.figure(figsize=(8, 5))
sns.histplot(df["BikeCount"], bins=30, kde=True)
plt.title("Distribution of Bike Rentals")
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Temperature", y="BikeCount")
plt.title("Temperature vs Bike Rentals")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Humidity", y="BikeCount")
plt.title("Humidity vs Bike Rentals")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Seasons", y="BikeCount",
            order=["Spring", "Summer", "Autumn", "Winter"])
plt.title("Bike Rentals by Season")
plt.show()

plt.figure(figsize=(10, 5))
avg_hour = df.groupby("Hour")["BikeCount"].mean()
plt.plot(avg_hour.index, avg_hour.values, marker="o")
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Rentals")
plt.show()

features = [
    "Hour",
    "Temperature",
    "Humidity",
    "WindSpeed",
    "Visibility",
    "DewPoint",
    "SolarRadiation",
    "Rainfall",
    "Snowfall",
    "Seasons",
    "Holiday",
    "FunctioningDay",
    "DayOfWeek",
    "Weekend",
    "PeakHour"
]

X = df[features]
y = df["BikeCount"]

cat_cols = ["Seasons", "Holiday", "FunctioningDay", "DayOfWeek"]
num_cols = [c for c in features if c not in cat_cols]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(pred[:200], label="Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()

residuals = y_test.values - pred

plt.figure(figsize=(7, 5))
plt.scatter(pred, residuals)
plt.axhline(0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Distribution")
plt.show()

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

prediction_df.to_csv("bike_predictions.csv", index=False)

print("\nPredictions saved to bike_predictions.csv")
print("Project completed.")
