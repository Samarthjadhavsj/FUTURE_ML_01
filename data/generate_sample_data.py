import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample retail sales data
np.random.seed(42)

# Date range: 2 years of daily data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Base sales with trend
base_sales = 1000
trend = np.linspace(0, 500, len(dates))

# Seasonality patterns
yearly_seasonality = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
weekly_seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)

# Random noise
noise = np.random.normal(0, 50, len(dates))

# Combine components
sales = base_sales + trend + yearly_seasonality + weekly_seasonality + noise

# Create dataframe
df = pd.DataFrame({
    'date': dates,
    'sales': sales.astype(int)
})

# Add some special events (spikes)
black_friday_2022 = df[df['date'] == '2022-11-25'].index
black_friday_2023 = df[df['date'] == '2023-11-24'].index
christmas_2022 = df[df['date'] == '2022-12-25'].index
christmas_2023 = df[df['date'] == '2023-12-25'].index

df.loc[black_friday_2022, 'sales'] *= 2.5
df.loc[black_friday_2023, 'sales'] *= 2.5
df.loc[christmas_2022, 'sales'] *= 2.0
df.loc[christmas_2023, 'sales'] *= 2.0

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print(f"Generated {len(df)} rows of sample sales data")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLast few rows:")
print(df.tail())
print(f"\nBasic statistics:")
print(df['sales'].describe())
