# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:31:44 2025

@author: user
"""

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

#Package
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Bootstrap Augmentation Function
def bootstrap_samples(X, y, n_samples=5000):
    """Create bootstrap samples with replacement"""
    indices = np.random.randint(0, len(X), size=(n_samples))
    if isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    X = X.reshape(-1, 1) if len(X.shape) == 1 else X
    return X[indices], y[indices]

# Read both datasets
file_path_2010 = r'D:\Research_2\Top 5 and TAIEX since 2010.xlsx'
file_path_00881 = r'D:\Research_2\00881 0050 TAIEX data.xlsx'

# Read data
df_2010 = pd.read_excel(file_path_2010, index_col='Date', parse_dates=True)
df_00881 = pd.read_excel(file_path_00881, index_col='Date', parse_dates=True)

# Set base date to start of 2010
base_date = '2010-01-01'
cutoff_date = '2020-12-10'

# Combine historical and actual data for Top 5
full_0050_data = pd.concat([
    df_2010['Top 5 TW'][df_2010.index < cutoff_date],
    df_00881['0050 TW'][df_00881.index >= cutoff_date]
])

# Get base prices (first available price in 2010)
base_price_0050 = full_0050_data.loc[full_0050_data.index >= base_date].iloc[0]

# Calculate returns for the actual data period (after cutoff_date)
returns_00881 = np.log(df_00881/df_00881.shift(1))
returns_00881 = returns_00881.dropna()

# Calculate historical returns for the period before cutoff_date
historical_data = df_2010[df_2010.index < cutoff_date].copy()
historical_returns = np.log(historical_data/historical_data.shift(1)).dropna()

# Prepare historical data for prediction
X_hist = historical_returns['Top 5 TW'].values.reshape(-1, 1)

# First, get the OLS regression coefficients using only actual data
X_recent = returns_00881['0050 TW']
y_recent = returns_00881['00881 TW']
X_recent = sm.add_constant(X_recent)
ols_model = sm.OLS(y_recent, X_recent).fit()

# Prepare data for Ridge regression
X = returns_00881['0050 TW'].values.reshape(-1, 1)
y = returns_00881['00881 TW'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Model (Without Bootstrap)
scaler_original = StandardScaler()
X_train_scaled_original = scaler_original.fit_transform(X_train)
X_hist_scaled = scaler_original.transform(X_hist)

ridge_model_original = Ridge(alpha=1.0)
ridge_model_original.fit(X_train_scaled_original, y_train)

# Apply bootstrap augmentation
X_boot, y_boot = bootstrap_samples(X_train, y_train, n_samples=len(X_train)*2)
X_train_aug = np.vstack([X_train, X_boot])
y_train_aug = np.concatenate([y_train, y_boot])

# Ridge Model with Bootstrap
scaler_boot = StandardScaler()
X_train_scaled_boot = scaler_boot.fit_transform(X_train_aug)
X_hist_scaled_boot = scaler_boot.transform(X_hist)

ridge_model_boot = Ridge(alpha=1.0)
ridge_model_boot.fit(X_train_scaled_boot, y_train_aug)

# Generate predictions using both models
hist_pred_original = ridge_model_original.predict(X_hist_scaled)
hist_pred_boot = ridge_model_boot.predict(X_hist_scaled_boot)

# Convert returns to price levels
start_price = df_00881['00881 TW'].iloc[0]  # First available price
hist_price_original = start_price * np.exp(hist_pred_original.cumsum())
hist_price_boot = start_price * np.exp(hist_pred_boot.cumsum())

# Create DataFrame with simulated historical prices
historical_prices = pd.DataFrame({
    'Date': historical_returns.index,
    'Simulated_00881_Original': hist_price_original,
    'Simulated_00881_Bootstrap': hist_price_boot
})

# For actual 00881 data (after cutoff date)
actual_00881 = df_00881['00881 TW']

# Calculate base prices for indexing
base_price_sim = hist_price_original[0]
base_scale = 100 / base_price_0050

# Calculate indexed prices
indexed_price_original = hist_price_original * base_scale
indexed_price_boot = hist_price_boot * base_scale
indexed_0050 = full_0050_data * base_scale
indexed_actual_00881 = actual_00881 * base_scale

# Set base date and get the starting value for base 100 indexing
base_date = '2010-01-01'
cutoff_date = '2020-12-10'

# Get the first value from 2010 for Top 5 series
base_value_top5 = df_2010.loc[df_2010.index >= base_date, 'Top 5 TW'].iloc[0]

# Index Top 5 series
indexed_top5 = (df_2010['Top 5 TW'] / base_value_top5) * 100

# Index the simulated series to start at 100 in 2010
indexed_sim_original = (hist_price_original / hist_price_original[0]) * 100
indexed_sim_boot = (hist_price_boot / hist_price_boot[0]) * 100

# Get the value of simulated series at cutoff date
sim_value_at_cutoff = indexed_sim_original[-1]  # Use the last value of simulation

# Index the actual 00881 series to match with simulation at cutoff date
first_actual_00881 = df_00881.loc[df_00881.index >= cutoff_date, '00881 TW'].iloc[0]
indexed_actual_00881 = (df_00881['00881 TW'] / first_actual_00881) * sim_value_at_cutoff

# Create the plot
plt.figure(figsize=(15, 8))

# Plot simulated data (before cutoff)
plt.plot(historical_returns.index, indexed_sim_original, color='tan',
         label='Ridge Simulated 00881 TW (Original)', linestyle='--')
plt.plot(historical_returns.index, indexed_sim_boot, color='crimson',
         label='Ridge Simulated 00881 TW (Bootstrap)', linestyle='--')

# Plot actual 00881 data (after cutoff)
plt.plot(df_00881.index, indexed_actual_00881, color='blue',
         label='Actual 00881 TW')

# Plot Top 5 data
plt.plot(indexed_top5.index, indexed_top5, color='darkgreen',
         label='Top 5 TW')

# Add cutoff line
plt.axvline(x=pd.to_datetime(cutoff_date), color='black', linestyle='--',
           label='Cutoff Date')

# Customize the plot
plt.title('00881 TW: Actual vs Simulated Prices (Base 100 = 2010)')
plt.xlabel('Date')
plt.ylabel('Indexed Price (Base 100 = 2010)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print base values for verification
print("\nBase Values (2010):")
print(f"Top 5 TW Base Value: {base_value_top5:.2f}")
print(f"Simulation Value at Cutoff: {sim_value_at_cutoff:.2f}")

# Create a DataFrame with all the indexed series, properly aligned by date
export_df = pd.DataFrame(index=historical_returns.index)
export_df['Ridge Simulated 00881 TW (Original)'] = indexed_sim_original
export_df['Ridge Simulated 00881 TW (Bootstrap)'] = indexed_sim_boot

# Add Top 5 TW data, matching dates
export_df = export_df.join(pd.DataFrame({'Top 5 TW': indexed_top5}))

# Add Actual 00881 TW data, matching dates
export_df = export_df.join(pd.DataFrame({'Actual 00881 TW': indexed_actual_00881}))

# Export to CSV
export_df.to_csv('indexed_prices_00881.csv')