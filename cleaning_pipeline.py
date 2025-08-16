import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import lightgbm as lgb
import numpy as np
from pathlib import Path

# 1. Load data
data = pd.read_csv("data/synthetic/synthetic_power_data.csv")

# 2. Convert timestamp to numerical features
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
data.drop('timestamp', axis=1, inplace=True)  # Remove original timestamp column

# 3. Outlier Detection (CLOF)
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
outliers = clf.fit_predict(data[["voltage", "current", "power"]])
data["is_outlier"] = outliers

# 4. Impute Missing Values (LightGBM)
for column in ["voltage", "current", "power"]:
    train_data = data[data[column] != -2]
    if len(train_data) > 0:
        # Convert to numpy arrays with proper dtype
        X = train_data.drop(columns=[column]).values.astype(np.float32)
        y = train_data[column].values.astype(np.float32)
        
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        
        missing = data[data[column] == -2]
        if len(missing) > 0:
            X_missing = missing.drop(columns=[column]).values.astype(np.float32)
            data.loc[data[column] == -2, column] = model.predict(X_missing)

# 5. Save cleaned data
Path("data/processed").mkdir(exist_ok=True)
data.to_csv("data/processed/cleaned_data.csv", index=False)
print("Data cleaning complete!")