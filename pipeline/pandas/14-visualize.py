#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Step 1: Remove the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Step 2: Rename the 'Timestamp' column to 'Date'
df = df.rename(columns={'Timestamp': 'Date'})

# Step 3: Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Step 4: Set 'Date' as the index
df = df.set_index('Date')

# Step 5: Fill missing values
df['Close'] = df['Close'].ffill()                             # Forward fill for 'Close' column  # noqa
df['High'] = df['High'].fillna(df['Close'])                   # Use 'Close' value for missing 'High'  # noqa
df['Low'] = df['Low'].fillna(df['Close'])                     # Use 'Close' value for missing 'Low'  # noqa
df['Open'] = df['Open'].fillna(df['Close'])                   # Use 'Close' value for missing 'Open'  # noqa
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)             # Set missing 'Volume_(BTC)' to 0  # noqa
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)   # Set missing 'Volume_(Currency)' to 0  # noqa

# Step 6: Group by day
df = df[df.index.year >= 2017]
df_plot = pd.DataFrame()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()

# Step 7: Print the Dataframe
print(df_plot)

# Step 8: Plot the data
df_plot.plot(x_compat=True)
plt.show()
