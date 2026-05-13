import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

#CONFIG
# Easy to change country for future practice
COUNTRY_CODE = 'USA'
COUNTRY_NAME = 'United States'

# LOAD DATA
print("=== Loading Dataset ===")
data = pd.read_csv('data/Indicators.bz2', compression='bz2')   # explicit compression (safer)
print("data.shape: ", data.shape)
print("Sample Data:\n", data.head(10))
print("Columns:\n", data.columns)

# Basic exploration
countries = data['CountryName'].unique().tolist()
print("Number of countries: ", len(countries))
countryCodes = data['CountryCode'].unique().tolist()
print("Number of country codes: ", len(countryCodes))
indicators = data['IndicatorName'].unique().tolist()
print("Number of indicators: ", len(indicators))
years = data['Year'].unique().tolist()
print("Number of years: ", len(years))
print("Year range:", min(years), "to", max(years))

#CO2 EMISSIONS ANALYSIS
print("\n=== CO2 Emissions Analysis for", COUNTRY_NAME, "===")
hist_indicator_co2 = 'CO2 emissions \(metric'
mask1 = data['IndicatorName'].str.contains(hist_indicator_co2)
mask2 = data['CountryCode'].str.contains(COUNTRY_CODE)
stage = data[mask1 & mask2]

print("Stage shape (CO2 data):", stage.shape)
print("Indicator Name:", stage['IndicatorName'].iloc[0])

# 1. Bar plot
plt.figure(figsize=(10, 6))
plt.bar(stage['Year'].values, stage['Value'].values, color='skyblue', edgecolor='navy')
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])
plt.title(f'CO2 Emissions per Capita in {COUNTRY_NAME} (Bar)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('co2_bar.png', dpi=200, bbox_inches='tight')
plt.show()

# 2. Line plot (cleaner & more appealing)
plt.figure(figsize=(10, 6))
plt.plot(stage['Year'].values, stage['Value'].values, marker='o', linewidth=2, color='darkgreen')
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])
plt.title(f'CO2 Emissions per Capita in {COUNTRY_NAME}')
plt.axis([1959, 2011, 0, 25])
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('co2_line.png', dpi=200, bbox_inches='tight')
plt.show()

# 3. Histogram of CO2 values for USA
hist_data = stage['Value'].values
print("Number of CO2 data points:", len(hist_data))
plt.figure(figsize=(10, 6))
plt.hist(hist_data, bins=10, density=False, facecolor='green', edgecolor='black', alpha=0.7)
plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Years')
plt.title(f'Histogram of CO2 Emissions per Capita in {COUNTRY_NAME}')
plt.grid(True)
plt.savefig('co2_histogram_usa.png', dpi=200, bbox_inches='tight')
plt.show()

# 4. Histogram of CO2 emissions for ALL countries in 2011 + USA annotation
hist_year = 2011
mask_year = data['Year'] == hist_year
co2_2011 = data[mask1 & mask_year]
print("Number of countries with CO2 data in 2011:", len(co2_2011))

plt.figure(figsize=(10, 6))
plt.hist(co2_2011['Value'], bins=10, density=False, facecolor='green', edgecolor='black', alpha=0.7)
plt.annotate("USA", xy=(18, 5), xycoords='data', xytext=(18, 30),
             textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Countries')
plt.title('Histogram of CO2 Emissions Per Capita (All Countries, 2011)')
plt.grid(True)
plt.savefig('co2_histogram_all_2011.png', dpi=200, bbox_inches='tight')
plt.show()

# GDP PER CAPITA ANALYSIS
print("\n=== GDP per Capita Analysis for", COUNTRY_NAME, "===")
hist_indicator_gdp = 'GDP per capita \(constant 2005'
mask1_gdp = data['IndicatorName'].str.contains(hist_indicator_gdp)
mask2 = data['CountryCode'].str.contains(COUNTRY_CODE)
gdp_stage = data[mask1_gdp & mask2]

print("GDP Stage shape:", gdp_stage.shape)
if not gdp_stage.empty:
    print("GDP Indicator Name:", gdp_stage['IndicatorName'].iloc[0])

# Line plot for GDP
plt.figure(figsize=(10, 6))
plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values, marker='o', linewidth=2, color='darkblue')
plt.xlabel('Year')
plt.ylabel(gdp_stage['IndicatorName'].iloc[0] if not gdp_stage.empty else 'GDP per capita')
plt.title(f'GDP per Capita in {COUNTRY_NAME}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('gdp_line.png', dpi=200, bbox_inches='tight')
plt.show()

# RELATIONSHIP: GDP vs CO2
print("\n=== Relationship between GDP and CO2 in", COUNTRY_NAME, "===")
print("GDP Min Year =", gdp_stage['Year'].min(), "Max:", gdp_stage['Year'].max())
print("CO2 Min Year =", stage['Year'].min(), "Max:", stage['Year'].max())

# Trim GDP to match CO2 years (original approach + small safety check)
gdp_stage_trunc = gdp_stage[gdp_stage['Year'] < 2012]
print("Length after trimming GDP:", len(gdp_stage_trunc))
print("Length of CO2 data:", len(stage))

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_title(f'CO2 Emissions vs. GDP per Capita in {COUNTRY_NAME}', fontsize=14)
ax.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[0])
ax.set_ylabel(stage['IndicatorName'].iloc[0])

X = gdp_stage_trunc['Value'].values
Y = stage['Value'].values[:len(X)]   # safety: ensure same length

ax.scatter(X, Y, color='purple', alpha=0.7, s=50)

# Add linear trend line + correlation in legend (this is the best improvement!)
if len(X) > 1:
    m, b = np.polyfit(X, Y, 1)
    correlation = np.corrcoef(X, Y)[0, 1]
    ax.plot(X, m * X + b, 'r--', linewidth=2, label=f'Trend line (Correlation = {correlation:.3f})')
    ax.legend(fontsize=12)

plt.savefig('gdp_vs_co2_scatter.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"Correlation value: {correlation:.3f}")
print("\n=== Analysis complete! Plots saved as PNG files. ===")
print("Key learning: After fixing the country mismatch, the real relationship between GDP and CO2 in the USA is now visible.")