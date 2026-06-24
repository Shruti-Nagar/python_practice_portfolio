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

