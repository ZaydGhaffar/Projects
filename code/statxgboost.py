
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

df = pd.read_csv('Clean_Data.csv')  
df['mls#'] = df['mls#'].astype(str)

mls_id = input("Enter MLS ID of the property: ")

house_data = df[df['mls#'] == mls_id][['date', 'price']]
house_data['date'] = pd.to_datetime(house_data['date'])
house_data.set_index('date', inplace=True)
house_data = house_data.asfreq('MS').fillna(method='ffill')
house_data['month_sin'] = np.sin(2 * np.pi * house_data.index.month / 12.0)

def add_noise(X_train, noise_level):
    noise = np.random.normal(scale=noise_level, size=X_train.shape)
    X_train_noisy = X_train + noise
    return X_train_noisy

def create_lags(dataset, n_lags):
    for lag in range(1, n_lags + 1):
        dataset[f'lag_{lag}'] = dataset['price'].shift(lag)
    return dataset.dropna()

def grid_search_parameters_with_early_stopping(data, lag_options, decay_options, early_stopping_rounds):
    best_mse = float('inf')
    best_lag = None
    best_decay = None
    
    for n_lags in lag_options:
        data_with_lags = create_lags(data.copy(), n_lags)
        X = data_with_lags.drop('price', axis=1)
        y = data_with_lags['price']
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            for decay_factor in decay_options:
                n = len(X_train)
                weights = np.exp(np.linspace(-decay_factor * n, 0, n))
                model = XGBRegressor(n_estimators=1000, learning_rate=0.05, objective='reg:squarederror')
                model.set_params(early_stopping_rounds=10)
                model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_test, y_test)], verbose=False)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                
                if mse < best_mse:
                    best_mse = mse
                    best_lag = n_lags
                    best_decay = decay_factor

    return best_lag, best_decay

lag_options = range(12, 61, 6)  
decay_options = np.linspace(0.01, 0.1, 10)  
best_lag, best_decay = grid_search_parameters_with_early_stopping(house_data[['price', 'month_sin']], lag_options, decay_options, early_stopping_rounds=10)

print(f"Optimal n_lags: {best_lag}, Optimal Decay Factor: {best_decay}")

house_data_with_lags = create_lags(house_data, best_lag)
X_final = house_data_with_lags.drop('price', axis=1)
y_final = house_data_with_lags['price']

n_final = len(X_final)
weights_final = np.exp(np.linspace(-best_decay * n_final, 0, n_final))

model_final = XGBRegressor(n_estimators=1000, learning_rate=0.05, objective='reg:squarederror')
model_final.set_params(early_stopping_rounds=10)

model_final.fit(X_final, y_final, sample_weight=weights_final, eval_set=[(X_final, y_final)], verbose=False)

noise_level = 0.5

X_final_noisy = add_noise(X_final, noise_level)

model_final.fit(X_final_noisy, y_final, sample_weight=weights_final, eval_set=[(X_final, y_final)], verbose=False)

def predict_future_prices_with_early_stopping(model, initial_features, steps):
    future_features = np.array(initial_features.copy())
    predictions = []

    for step in range(steps):
        month_feature = np.sin(2 * np.pi * ((step % 12) + 1) / 12.0)
        future_features[-1] = month_feature 
        
        pred = model.predict(future_features.reshape(1,-1))[0]
        predictions.append(pred)
        
        future_features = np.roll(future_features, -1)
        future_features[-2] = pred  
    
    return predictions

initial_features = X_final.iloc[-1].values

predictions_12_months = predict_future_prices_with_early_stopping(model_final, initial_features, 12)
predictions_36_months = predict_future_prices_with_early_stopping(model_final, initial_features, 36)
predictions_60_months = predict_future_prices_with_early_stopping(model_final, initial_features, 60)

print(f"Predicted price in 1 year for MLS# {mls_id}: ${predictions_12_months[-1]:.2f}")
print(f"Predicted price in 3 years for MLS# {mls_id}: ${predictions_36_months[-1]:.2f}")
print(f"Predicted price in 5 years for MLS# {mls_id}: ${predictions_60_months[-1]:.2f}")
future_dates = pd.date_range(start=house_data.index[-1] + pd.DateOffset(months=1), periods=60, freq='MS')
predictions_series = pd.Series(data=predictions_60_months, index=future_dates)

plt.figure(figsize=(10, 6))
plt.plot(house_data['price'], label='Historical Prices', color='blue')

plt.plot(predictions_series, label='Predicted Prices', color='red', linestyle='--')

plt.title(f"Real Estate Price Trend and Forecast for MLS# {mls_id}")
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()