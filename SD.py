# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("C:/Users/LENOVO/OneDrive/Desktop/SD.csv")  # Update with your dataset file name and path

# Split data into features (X) and target variable (y)
X = data.drop('ElectricMotor', axis=1)  # Update 'ElectricMotor' with your target column name
y = data['ElectricMotor']  # Update 'ElectricMotor' with your target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Example usage: Predicting the electric motor for a new boat

new_boat = pd.DataFrame({
    # Update with your new boat's specifications
    'Seating Capacity': [100],
    'Length': [18.4],
    'Breadth':[4.58],
    'Height':[2.78],
    'Weight':[12746],
    'CC':[150],
    'Speed':[20],
})
new_boat_scaled = scaler.transform(new_boat)
predicted_motor = model.predict(new_boat_scaled)
print(f"Predicted Electric Motor: {predicted_motor[0]}")
