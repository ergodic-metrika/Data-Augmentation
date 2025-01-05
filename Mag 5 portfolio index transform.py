# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:01:56 2025

@author: user
"""
#Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read excel file
file_path = r'D:\Research_2\cleaned_Mag 5 TAIEX.xlsx'
df = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# Create the weighted index
weights = {
    'NVDA': 0.20,
    'AVGO': 0.20,
    'AAPL': 0.20,
    'MSFT': 0.20,
    'NFLX': 0.20
}

# Calculate weighted index (starting at 100)
df['Mag5_Index'] = 100 * (
    weights['NVDA'] * df['NVDA']/df['NVDA'].iloc[0] +
    weights['AVGO'] * df['AVGO']/df['AVGO'].iloc[0] +
    weights['AAPL'] * df['AAPL']/df['AAPL'].iloc[0] +
    weights['MSFT'] * df['MSFT']/df['MSFT'].iloc[0] +
    weights['NFLX'] * df['NFLX']/df['NFLX'].iloc[0]
)

# Plot the results
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Mag5_Index'], label='Mag5 Index')
plt.plot(df.index, df['TAIEX']/df['TAIEX'].iloc[0] * 100, label='TAIEX')

plt.title('Mag5 Index vs TAIEX (Base 100)')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Export to Excel
df.to_excel('Mag5_Index.xlsx')


