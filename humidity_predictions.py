import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


file_path=open("sensor_data.csv","r")
df= pd.read_csv(file_path)
corrected_timestamp = df["timestamp"] - (795666791 - 1710801452)  # Adjust based on actual reference
df["datetime"] = pd.to_datetime(corrected_timestamp, unit="s", utc=True)
df["datetime"] = df["datetime"].apply(lambda x: x.replace(year=2025) if x.year == 2024 else x)
print(df)

print(df)
df_cleaned = df.drop_duplicates(subset=["temperature", "humidity"], keep="first")
print(df_cleaned)

df.describe()

# Ensure timestamps are correctly converted
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Use "ms" if necessary
df["timestamp"] = df["timestamp"] + pd.DateOffset(years=30)  # Adjust incorrect years

# Standardize time for ML models
scaler = StandardScaler()
X = scaler.fit_transform(df["timestamp"].astype("int64").values.reshape(-1, 1) // 10**9)  # Convert to seconds
y = df["temperature"].values

# Fit Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Fit Polynomial Regression
poly_degree = 5
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model.fit(X, y)

# Generate smooth polynomial predictions
X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_poly_pred = poly_model.predict(X_smooth)

# Compute Least Squares Solution using Normal Equation
X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
y_normal_eq_pred = X_b @ theta_best

# Singular Value Decomposition (SVD) for Matrix Factorization
U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
S_inv = np.diag(1 / S)
theta_svd = Vt.T @ S_inv @ U.T @ y
y_svd_pred = X_b @ theta_svd

# Convert scaled X values back to timestamps
X_smooth_dates = pd.to_datetime(scaler.inverse_transform(X_smooth).flatten().astype("int64"), unit="s")

# Plot results
plt.figure(figsize=(12, 5))
plt.scatter(df["timestamp"], df["temperature"], label="Actual Data", s=5, alpha=0.5)
plt.plot(df["timestamp"], y_linear_pred, label="Linear Regression", color="red", linewidth=2)
plt.plot(X_smooth_dates, y_poly_pred, label=f"Polynomial Regression (Degree {poly_degree})", color="green", linewidth=2)
plt.plot(df["timestamp"], y_normal_eq_pred, label="Normal Equation", color="blue", linestyle="dashed")
plt.plot(df["timestamp"], y_svd_pred, label="SVD Regression", color="purple", linestyle="dotted")

# Format x-axis
plt.xlabel("Date & Time")
plt.ylabel("Temperature (°C)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45, ha="right")

plt.legend()
plt.title("Temperature Trend Over Time (Optimized Polynomial Fit)")
plt.tight_layout()
plt.savefig("Temperature Trend Over Time.png")

# Newton's Method for finding optimal heating/cooling thresholds
def newtons_method(temps, initial_threshold=22, tol=1e-6, max_iter=100):
    threshold = initial_threshold
    for _ in range(max_iter):
        gradient = sum(2 * (temp - threshold) for temp in temps)
        hessian = 2 * len(temps)
        step = gradient / hessian
        if abs(step) < tol:
            break
        threshold -= step
    return threshold

# Calculate optimal threshold using Newton's Method
temp_series = df["temperature"].values
optimal_temp = newtons_method(temp_series)
print(f"Optimal heating/cooling threshold (Newton's Method): {optimal_temp}")

# Iterative function to find the best heating/cooling threshold
def optimal_threshold(temps, heating_threshold=20, cooling_threshold=25):
    cost = 0
    for temp in temps:
        if temp < heating_threshold:
            cost += (heating_threshold - temp) ** 2
        elif temp > cooling_threshold:
            cost += (temp - cooling_threshold) ** 2
    return cost

# Calculate cost for given thresholds
total_cost = optimal_threshold(temp_series)
print(f"Total deviation cost: {total_cost}")

# humidty over time plot


# Standardize time for ML models
scaler = StandardScaler()
X = scaler.fit_transform(df["timestamp"].astype("int64").values.reshape(-1, 1) // 10**9)  # Convert to seconds
y = df["humidity"].values

# Fit Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Fit Polynomial Regression
poly_degree = 5
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model.fit(X, y)

# Generate smooth polynomial predictions
X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_poly_pred = poly_model.predict(X_smooth)

# Compute Least Squares Solution using Normal Equation
X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
y_normal_eq_pred = X_b @ theta_best

# Singular Value Decomposition (SVD) for Matrix Factorization
U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
S_inv = np.diag(1 / S)
theta_svd = Vt.T @ S_inv @ U.T @ y
y_svd_pred = X_b @ theta_svd

# Convert scaled X values back to timestamps
X_smooth_dates = pd.to_datetime(scaler.inverse_transform(X_smooth).flatten().astype("int64"), unit="s")

# Plot results
plt.figure(figsize=(12, 5))
plt.scatter(df["timestamp"], df["humidity"], label="Actual Data", s=5, alpha=0.5)
plt.plot(df["timestamp"], y_linear_pred, label="Linear Regression", color="red", linewidth=2)
plt.plot(X_smooth_dates, y_poly_pred, label=f"Polynomial Regression (Degree {poly_degree})", color="green", linewidth=2)
plt.plot(df["timestamp"], y_normal_eq_pred, label="Normal Equation", color="blue", linestyle="dashed")
plt.plot(df["timestamp"], y_svd_pred, label="SVD Regression", color="purple", linestyle="dotted")

# Format x-axis
plt.xlabel("Date & Time")
plt.ylabel("Humidty (%)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45, ha="right")

plt.legend()
plt.title("Humdity Trend Over Time (Optimized Polynomial Fit)")
plt.tight_layout()
plt.savefig("Humdity Trend Over Time.png")