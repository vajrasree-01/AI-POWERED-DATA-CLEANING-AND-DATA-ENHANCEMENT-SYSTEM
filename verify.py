import pandas as pd
data = pd.read_csv("data/synthetic/synthetic_power_data.csv")
print(data.head())
print("\nMissing values (current = -2):", data[data["current"] == -2].shape[0])
print("Voltage outliers (>300V):", data[data["voltage"] > 300].shape[0])