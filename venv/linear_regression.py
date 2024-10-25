import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with correct column names
data = pd.read_csv(r'D:\third year\3.2\AI\pythonMl\Office_Price.csv')
x = data['SIZE'].values  # Correct column for office size
y = data['PRICE'].values  # Correct column for office price

# Define Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c

    # Calculate gradients
    dm = -(2/N) * sum(x * (y - y_pred))
    dc = -(2/N) * sum(y - y_pred)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    return m, c

# Initial parameters
m = np.random.randn()
c = np.random.randn()
epochs = 10
learning_rate = 0.01

# Run gradient descent for the specified epochs
for epoch in range(epochs):
    m, c = gradient_descent(x, y, m, c, learning_rate)
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f'Epoch {epoch+1}, MSE: {error:.4f}')

# Plot data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plot line of best fit
plt.plot(x, m * x + c, color='red', label='Best Fit Line')

# Labeling the plot
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.legend()
plt.show()

# Predict price for an office of 100 sq. ft.
office_size = 100
predicted_price = m * office_size + c
print(f'Predicted price for a 100 sq. ft. office: {predicted_price:.2f}')
