# Exploring and analysis of data before Pre processing

import pandas as pd
import numpy as np

# load dataset weather_data
df = pd.read_csv('weather_data.csv', parse_dates=['date'])

print('Shape & Data Types')
print(f'Rows: {df.shape[0]}')
print(f'Cols: {df.shape[1]}')
print('\nColumn names & dtypes:')
print(df.dtypes.to_string())

print('# first and last 5 rows')
print('\nhead:')
print(df.head().to_string(index=False))
print('\ntail:')
print(df.tail().to_string(index=False))
print('\n')

print('# descriptive statistics:')
print(df.describe().round(2).to_string())
print('\n')

print('# missing value audit:')
missing_cnt = df.isnull().sum()
missing_pct = (missing_cnt/len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'missing_cnt': missing_cnt,
    'missing_pct_%': missing_pct
})
print(missing_report.to_string())
print('\n')

print('target variable distribution')
target_cnts = df['will_rain_tomorrow'].value_counts()
target_pct = df['will_rain_tomorrow'].value_counts(normalize=True).mul(100).round(2)
print('(0 = No Rain Tomorrow | 1 = Rain Tomorrow)')
print('\n')
print('# Correlation with Target Variable')
numeric_cols = ['temperature_c', 
                'humidity_pct', 
                'wind_speed_kmh', 
                'precipitation_mm', 
                'cloud_cover_pct',
                'pressure_hpa']
correlations = df[numeric_cols].corrwith(df['will_rain_tomorrow'])
print(correlations.sort_values(ascending=False).to_string())
print('\n')

print('# Monthly average temperature (sanity check)')
monthly_avg = (
    df.groupby('month')['temperature_c'].mean().round(1).rename('avg_temp_c')
)
print(monthly_avg.to_string())
