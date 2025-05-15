```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('silver_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
prices = data['Price'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Prepare data for LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_prices, time_step)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][time_step:train_size+time_step], train_predict, label='Train Predictions')
plt.plot(data['Date'][train_size+time_step:], test_predict, label='Test Predictions')
plt.plot(data['Date'][time_step:train_size+time_step], y_train_inv.T, label='Actual Train')
plt.plot(data['Date'][train_size+time_step:], y_test_inv.T, label='Actual Test')
plt.title('Silver Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('silver_price_prediction.png')
plt.show()

# Predict future prices (next 30 days)
last_60_days = scaled_prices[-time_step:]
future_predictions = []
for _ in range(30):
    X_pred = last_60_days.reshape(1, time_step, 1)
    pred = model.predict(X_pred, verbose=0)
    future_predictions.append(pred[0, 0])
    last_60_days = np.append(last_60_days[1:], pred, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Save future predictions
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions.flatten()})
future_df.to_csv('future_silver_predictions.csv', index=False)
print("Future predictions saved to 'future_silver_predictions.csv'.")
```
