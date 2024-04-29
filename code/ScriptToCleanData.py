import pandas as pd

df = pd.read_csv('zestimate_history_unclean.csv')

df.drop(' Timestamp170619808', axis=1, inplace=True)

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

filtered_df = df.groupby('mls#').filter(lambda x: len(x) > 12)

filtered_df.to_csv('Clean_Data.csv', index=False)
