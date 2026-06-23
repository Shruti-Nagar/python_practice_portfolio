import pandas as pd
import numpy as np

# reproducibility
np.random.seed(42)

# 1. building data range
dates = pd.date_range(
    start='2021-01-01', 
    end='2023-12-31', 
    freq='D')
n = len(dates)

# 2. date feature arrays
day_of_year = dates.day_of_year
month = dates.month

# sine wave centered at summer peak (day - 172 [june 21])
seasonal_signal = np.sin(2 * np.pi * (day_of_year - 80) / 365)
# -- Amplitude = maximum distance from average to peak/lowest point
# 3. Temperature (°C)
# base: oscillation between -7°C and 35°C degrees celcius
# temperature = amplitude + baseline(avg) * (+1/-1)
temp_base = 21 + 14 * seasonal_signal

# add realistic day-to-day random noise (std = 3 degrees celcius)
temp_noise = np.random.normal(
    loc = 0, 
    scale = 3, 
    size = n)
temperature = np.round(temp_base + temp_noise, 1)

# 4. Humiduty (%)
# Humidity is inversely correlated with temperature
# humidity varying between 50% and 80% ~avg 65 
# amplitude 65-50 , 80-65 = 15
humidity_base = 65 - 15 * seasonal_signal
humidity_noise = np.random.normal(
    loc = 0, 
    scale = 8, 
    size = n)
humidity = np.clip(np.round(humidity_base + humidity_noise, 1), 10,100)

# 5. Wind Speed (km/h)
# gamma distribution?
wind_speed = np.round(np.random.gamma(
    shape=2.0, 
    scale = 7.5, 
    size = n),1)

# 6. precipitation (mm)
# mostdays = 0 mm; rainy days follow an exponential distribution
rain_mask = np.random.random(size=n) < 0.30 #30% chance of rain
precipitation = np.where(rain_mask, 
                        np.round(np.random.exponential(
                            scale=8.0, 
                            size=n), 1), 
                        0.0)

# 7. Cloud Cover (%)
# correlated with precipitation: more cloud on rainy days
# baseline + extra cloud cover during rain, precipitation between (0-1)
cloud_base = 50 + 20 * (precipitation > 0).astype(float)
cloud_noise = np.random.normal(
    loc=0, 
    scale=15, 
    size=n)
cloud_cover = np.clip(np.round(cloud_base + cloud_noise, 1), 0, 100)

# 8. Pressure (hPa)
# realistic sea level pressure is normally distriuted around 1013 hPa
pressure = np.round(np.random.normal(
    loc=1013, 
    scale=8, 
    size=n),1 )

# 9. Target variable: Will it rain tomorrow?
# Shift precipitation by -1 day; last day gets NaN - drop it
will_rain_tomorrow = (pd.Series(precipitation).shift(-1) > 0).astype(float)

# 10. Assemble DataFrame
df = pd.DataFrame({
    'date': dates,
    'month': month,
    'day_of_year': day_of_year,
    'temperature_c': temperature,
    'humidity_pct': humidity,
    'wind_speed_kmh': wind_speed,
    'precipitation_mm': precipitation,
    'cloud_cover_pct': cloud_cover,
    'pressure_hpa': pressure,
    'will_rain_tomorrow': will_rain_tomorrow
})

# drop the last row (NaN target from the shift)
df = df.dropna(subset = ['will_rain_tomorrow']).reset_index(drop=True)
df['will_rain_tomorrow'] = df['will_rain_tomorrow'].astype(int)

# introduce realistic missing values
# sensor outages -> randomly blank -2% of wind speed and pressure
for col in ['wind_speed_kmh', 'pressure_hpa']:
    missing_idx = np.random.choice(
        df.index, 
        size = int(0.02*len(df)), 
        replace = False)
    df.loc[missing_idx, col] = np.nan

# Save data and preview
df.to_csv(r'C:\Users\DELL\Desktop\weather app\weather_data.csv', index=False)
print(f'Dataset saved -> {len(df)} rows x {df.shape[1]} columns\n')
print(df.head(100).to_string(index=False))
