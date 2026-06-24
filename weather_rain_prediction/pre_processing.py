import pandas as pd
import numpy as np

df=pd.read_csv('weather_data.csv', parse_dates = ['date'])

print('# Missing Value Audit\n')
print(df.isnull().sum().to_string())

# Median Imputation for wind_speed_kmh
wind_median = df['wind_speed_kmh'].median()

df['wind_speed_kmh'] = df['wind_speed_kmh'].fillna(wind_median)
print(f'\nwind_speed_kmh median = {wind_median:.2f} km/h')

# Median Imputation for pressure_hpa
pressure_median = df['pressure_hpa'].median()

df['pressure_hpa'] = df['pressure_hpa'].fillna(pressure_median)
print(f'pressure_hpa median = {pressure_median:.2f} hpa')

print('\n# after imputation -- missing value audit')
print(df.isnull().sum().to_string())

#FEATURE ENGINEERING

# CYCLICAL ENCODING
# sine/cosine encoding mapes time series onto a circle so that 
# distance reflects true calender proximity
df['month_sin'] = np.sin(2*np.pi * df['month']/12)
df['month_cos'] = np.cos(2*np.pi * df['month']/12)

# cyclical encoding for day_of_year
df['doy_sin'] = np.sin(2*np.pi * df['day_of_year']/365)
df['doy_cos'] = np.sin(2*np.pi * df['day_of_year']/365)

# Magnus-Tetens Heat Index Approximation - to calc heat index
df['heat_index'] = np.round(
    df['temperature_c'] +
    0.33 * (df['humidity_pct']/100
            * 6.105 * np.exp((17.27 * df['temperature_c'])/ 
                             (237.7 * df['temperature_c'])))
                             - 4.0, 2)

# Calc 3-day avg precipitation
df['precip_3day_avg'] = (
    df['precipitation_mm'].rolling(
        window=3, 
        min_periods=1)
        .mean().round(2)
    )

# Calc 3-day avg cloud cover
df['cloud_3day_avg'] = (
    df['cloud_cover_pct'].rolling(
        window=3, 
        min_periods=1).mean().round(2)
)

# pressure change from previous day
# So.... a falling barometer(neg diff) strongly signals incoming rain.
# this is one of the most powerful meteorological signals.
df['pressure_change'] = df['pressure.hps'].diff().round(2)

# fill 1st day with 0.0 as it doesn't have any previous day
df['pressure_change'] = df['pressure_change'].fillna(0.0)

# binary rain flag for today
df['rained_today'] = (df['precipitation_mm'] > 0).astype(int)

new_cols = ['month_sin', 'month_cos', 'doy_sin', 'doy_cos', 
            'heat_index', 'precip_3day_avg', 'cloud_3day_avg',
            'pessure_change', 'rained_today']
print(f'\n{len(new_cols)} New Features added:')

for c in new_cols:
    print(c)

print(f'\nDataset now has {df.shape[1]} columns and {df.shape[0]} rows.')

# print(df.head())

# FEATURE SCALING
