# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:50:49 2025

@author: user
"""

#Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read excel file
file_path = r'D:\Research_2\Mag 5 TAIEX.xlsx'
df = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# Print initial missing values
print("\nInitial Missing Values:")
print(df.isnull().sum())

# Fill TAIEX missing values
df['TAIEX'] = df['TAIEX'].fillna(method='ffill')

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Export cleaned data
df.to_excel('cleaned_Mag 5 TAIEX.xlsx')

print("\nCleaned data has been exported to Excel")


#Then remake an index with
#20% weight for NVDA
#20% weight for AVGO
#20% weight for AAPL
#20% weight for MSFT
#20% weight for NFLX

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


