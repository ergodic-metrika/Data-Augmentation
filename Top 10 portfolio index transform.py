# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:25:24 2025

@author: user
"""

#Package
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Read excel file
file_path = r'D:\Research_2\Top 10.xlsx'
df = pd.read_excel(file_path, index_col='Date', parse_dates=True)


# Create weights dictionary
weights = {
    '2330 TW': 0.3997,
    '2317 TW': 0.1632,
    '2454 TW': 0.1537,
    '2308 TW': 0.06509,
    '2382 TW': 0.05930,
    '2303 TW': 0.0433,
    '2345 TW': 0.0380,
    '3231 TW': 0.0262,
    '3034 TW': 0.0261,
    '3008 TW': 0.0252
}

# Calculate returns
returns = df.pct_change()

# Calculate weighted portfolio returns
portfolio_returns = pd.Series(0, index=returns.index)
for stock, weight in weights.items():
    portfolio_returns += returns[stock] * weight

# Create index series with base 100
portfolio_index = pd.Series(100 * (1 + portfolio_returns).cumprod(), index=returns.index)
portfolio_index.name = 'Index'

# Create DataFrame with date as first column and index as second column
index_df = pd.DataFrame(portfolio_index).reset_index()
index_df.columns = ['Date', 'Index']

# Export to Excel
excel_path = r'D:\Research_2\Top 10 Portfolio_Index.xlsx'
index_df.to_excel(excel_path, index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(index_df['Date'], index_df['Index'])
plt.title('Top 10 Portfolio Index (Base 100)')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.grid(True)