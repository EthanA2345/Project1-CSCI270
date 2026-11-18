
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Aotizhongxin monitoring station
df = pd.read_csv(
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

# Combine date columns into a datetime index
df["datetime"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

df = df.set_index("datetime")

# Isolate PM2.5 measurements
pm25 = df["PM2.5"]


# Investigate consecutive missing PM2.5 measurements

missing = pm25.isna()

# Identify groups of consecutive missing values
groups = (missing != missing.shift()).cumsum()

# Calculate the length of each missing-data gap
missing_runs = missing[missing].groupby(groups[missing]).size()

print("\nMISSING DATA GAP ANALYSIS")
print("Number of separate missing gaps:", len(missing_runs))
print("Longest missing gap:", missing_runs.max(), "hours")
print("Average missing gap:", round(missing_runs.mean(), 2), "hours")

print("\nTen longest missing gaps:")
print(missing_runs.sort_values(ascending=False).head(10))

# Categorize missing gaps by length
print("\nGAP LENGTH DISTRIBUTION")
print("1-6 hours:", ((missing_runs >= 1) & (missing_runs <= 6)).sum())
print("7-24 hours:", ((missing_runs >= 7) & (missing_runs <= 24)).sum())
print("25-72 hours:", ((missing_runs >= 25) & (missing_runs <= 72)).sum())
print("Over 72 hours:", (missing_runs > 72).sum())


# Check for missing values
print("Total observations:", len(pm25))
print("Missing PM2.5 values:", pm25.isna().sum())

# Handle missing values using linear interpolation  

# Handle missing PM2.5 measurements

# Identify missing values
missing = pm25.isna()

# Identify groups of consecutive missing values
groups = (missing != missing.shift()).cumsum()

# Calculate the length of each missing gap
gap_lengths = missing.groupby(groups).transform("sum")

# Identify missing gaps of 6 hours or shorter
short_missing = missing & (gap_lengths <= 6)

# Perform linear interpolation using timestamps
interpolated = pm25.interpolate(
    method="time",
    limit_area="inside"
)

# Fill only short gaps
pm25_clean = pm25.copy()
pm25_clean.loc[short_missing] = interpolated.loc[short_missing]

# Display correction results
print("\nMISSING DATA CORRECTION")
print("Values interpolated:", short_missing.sum())
print("Values still missing:", pm25_clean.isna().sum())


# Resample hourly measurements into daily averages

# Calculate daily PM2.5 averages

daily_pm25 = pm25_clean.resample("D").mean()

# Count valid hourly measurements per day
daily_counts = pm25.resample("D").count()

# Require at least 18 valid hourly measurements
daily_pm25[daily_counts < 18] = np.nan

print("\nDAILY AVERAGE ANALYSIS")
print("Total days:", len(daily_pm25))
print("Days with valid averages:", daily_pm25.notna().sum())
print("Days without valid averages:", daily_pm25.isna().sum())


# Fit a linear trend line
valid_daily = daily_pm25.dropna()


# Fit a linear trend using actual elapsed calendar days

valid_daily = daily_pm25.dropna()

# Calculate days since the first valid observation
x = (valid_daily.index - valid_daily.index[0]).days
y = valid_daily.values

# Calculate best-fit line
slope, intercept = np.polyfit(x, y, 1)

trend = slope * x + intercept

# Calculate annual change
annual_change = slope * 365.25

print("\nCORRECTED TREND ANALYSIS")
print("Daily trend:", round(slope, 6), "µg/m³ per day")
print("Estimated annual change:", round(annual_change, 4), "µg/m³ per year")


# Plot daily averages and trend line
plt.figure(figsize=(12, 6))

plt.scatter(
    valid_daily.index,
    valid_daily.values,
    s=8,
    alpha=0.4,
    label="Daily PM2.5 averages"
)

plt.plot(
    valid_daily.index,
    trend,
    color="red",
    linewidth=2,
    label="Linear trend line"
)

plt.title("Daily PM2.5 Concentration at Aotizhongxin")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("PM25_Corrected_Graph.png", dpi=300)
plt.show()
