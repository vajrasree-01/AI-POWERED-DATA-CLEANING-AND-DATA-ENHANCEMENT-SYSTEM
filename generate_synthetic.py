import pandas as pd
import numpy as np
import os
from pathlib import Path

# 1. Create required directories if they don't exist
data_dir = Path("data/synthetic")
plots_dir = Path("plots")
data_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# 2. Generate synthetic data with updated frequency
np.random.seed(42)
timestamps = pd.date_range("2023-01-01", periods=1000, freq="15min")  # Using 'min' instead of 'T'
data = pd.DataFrame({
    "timestamp": timestamps,
    "voltage": np.random.normal(220, 10, 1000),
    "current": np.random.normal(15, 3, 1000),
    "power": np.random.normal(3000, 500, 1000)
})

# 3. Inject anomalies
data.loc[10:15, "current"] = -2  # Missing values
data.loc[100:105, "voltage"] = 400  # Outliers

# 4. Save data with absolute path
output_path = data_dir / "synthetic_power_data.csv"
data.to_csv(output_path, index=False)
print(f"Data successfully generated at: {output_path.absolute()}")