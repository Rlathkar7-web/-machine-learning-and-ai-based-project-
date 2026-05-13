import folium
import pandas as pd
import warnings
import webbrowser
from pathlib import Path

warnings.filterwarnings("ignore")

# CONFIG
COUNTRY_GEO = 'data/world-countries.json'
DATA_FILE = 'data/Indicators.bz2'

# Change these to experiment with different years/indicators
HIST_INDICATOR_PARTIAL = 'CO2 emissions \(metric'   # matches "CO2 emissions (metric tons per capita)"
HIST_YEAR = 1980

OUTPUT_DIR = Path('saved_info')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f'co2_emissions_{HIST_YEAR}.html'
#
print("🔄 Loading World Development Indicators dataset...")
data = pd.read_csv(DATA_FILE, compression='bz2')
print(f"✅ Data Shape: {data.shape}")
print(f"Sample Data:\n{data.head()}")

# FILTER DATA
print(f"\n🔍 Filtering for '{HIST_INDICATOR_PARTIAL}' in year {HIST_YEAR}...")
mask1 = data['IndicatorName'].str.contains(HIST_INDICATOR_PARTIAL, regex=True)
mask2 = data['Year'] == HIST_YEAR
stage = data[mask1 & mask2].copy()

if stage.empty:
    raise ValueError(f"No data found for indicator containing '{HIST_INDICATOR_PARTIAL}' in {HIST_YEAR}!")

print(f"✅ Selected {len(stage)} countries with data.")

# Prepare data for choropleth (CountryCode must match GeoJSON feature.id)
plot_data = stage[['CountryCode', 'Value']]
hist_indicator_full = stage.iloc[0]['IndicatorName']
print(f"📊 Indicator used: {hist_indicator_full}")

# CREATE INTERACTIVE MAP
print("🗺️  Building Folium map...")

m = folium.Map(
    location=[20, 0],          # sensible world center (latitude, longitude)
    tiles="Cartodb Positron",  # clean, professional tiles for choropleth
    zoom_start=2,
    attr='Cartodb Positron'
)

# Modern way (choropleth method is deprecated)
folium.Choropleth(
    geo_data=COUNTRY_GEO,
    name="CO₂ Emissions",
    data=plot_data,
    columns=['CountryCode', 'Value'],
    key_on='feature.id',           # must match your world-countries.json structure
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=hist_indicator_full,
    highlight=True
)