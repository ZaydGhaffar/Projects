from neuralprophet import NeuralProphet
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('Clean_Data.csv')
df['mls#'] = df['mls#'].astype(str)
mls_id = '170619808' #415k is the current market price and the 1 and 5 year predictons are 391k and 576k
# mls_id = '170624394' #305000 is the current market price and the 1 and 5 year predictons are 391k and 623k
# mls_id = '170623228' #339000 is the current market price and the 1 and 5 year predictons are $496519.50 and  $623642.97
# mls_id = '170619797' #350000 is the current market price and the 1 and 5 year predictons are 112k and 131k - abrupt change
# mls_id = '170588674' #750k is the current market price and the 1 and 5 year predictons are 730k and 630k
# mls_id = '170588674' #750000
house_data = df[df['mls#'] == mls_id][['date', 'price']]
house_data.rename(columns={'date': 'ds', 'price': 'y'}, inplace=True)
house_data['ds'] = pd.to_datetime(house_data['ds'])

model = NeuralProphet()

metrics = model.fit(house_data, freq='M')  

future = model.make_future_dataframe(house_data, periods=60)  

forecast = model.predict(future)

price_1_year = forecast.iloc[12]['yhat1']  
price_5_years = forecast.iloc[59]['yhat1'] 

print("Current Price of MLS# 170619808 = 415,000")
print(f"Predicted price in 1 year for MLS# {mls_id}: ${price_1_year:.2f}")
print(f"Predicted price in 5 years for MLS# {mls_id}: ${price_5_years:.2f}")

fig_forecast = model.plot(forecast)
fig_forecast.show()