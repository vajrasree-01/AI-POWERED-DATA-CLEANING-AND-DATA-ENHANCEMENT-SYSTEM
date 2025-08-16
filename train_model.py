import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import joblib

# Load cleaned data
data = pd.read_csv("data/processed/cleaned_data.csv")

# Prepare features (X) and target (y)
X = data.drop(columns=["power"])  
y = data["power"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM model with optimized parameters
model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    learning_rate=0.05,  # Reduce learning rate for better convergence
    n_estimators=500,     # Increase estimators for more learning
    max_depth=5,          # Limit depth to avoid overfitting
    num_leaves=20,        # Set reasonable number of leaves
    min_child_samples=10  # Prevent overfitting by requiring min 10 samples per leaf
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions) ** 0.5  # Manual RMSE calculation
print(f"RMSE: {rmse:.2f}")

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/power_predictor.pkl")
print("Model saved successfully!")
