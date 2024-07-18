import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import seaborn as sns

file_paths = {
    'fire': 'C:\\Users\\gian\\Documents\\Sacred Codes\\fire.csv',
    'earthquake': 'C:\\Users\\gian\\Documents\\Sacred Codes\\earthquake.csv',
    'flood': 'C:\\Users\\gian\\Documents\\Sacred Codes\\flood.csv'
}

disaster_dataframes = []
for disaster_type, file_path in file_paths.items():
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    df['type'] = disaster_type
    disaster_dataframes.append(df)

disaster_data = pd.concat(disaster_dataframes, ignore_index=True)

disaster_data = disaster_data[disaster_data['date'] != "0"]

disaster_data['date'] = pd.to_datetime(disaster_data['date'], errors='coerce')

disaster_data = disaster_data.dropna(subset=['date'])

if 'damages' in disaster_data.columns:
    disaster_data['damages'] = disaster_data['damages'].replace('[\D]', '', regex=True).astype(float)
disaster_data.fillna(0, inplace=True)

disaster_data = disaster_data[(np.abs(zscore(disaster_data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

summary_statistics = disaster_data.describe()

disaster_counts = disaster_data['type'].value_counts()

disaster_data['year'] = disaster_data['date'].dt.year
disaster_data['year_month'] = disaster_data['date'].dt.to_period('M')
monthly_incidents = disaster_data.groupby('year_month').size()

total_incidents = len(disaster_data)
total_years = disaster_data['date'].dt.year.nunique()
incidents_per_year = total_incidents / total_years

threshold = 10

is_prone_to_disasters = incidents_per_year > threshold

model = ARIMA(monthly_incidents, order=(1, 1, 1))
fit_model = model.fit()

forecast = fit_model.forecast(steps=12)

yearly_incidents = disaster_data.groupby('year').size()
year_with_most_disasters = yearly_incidents.idxmax()
most_disasters_count = yearly_incidents.max()

earthquake_data = disaster_data[disaster_data['type'] == 'earthquake']

if not earthquake_data.empty:
    plt.figure(figsize=(10, 5))
    earthquake_data['magnitude'].plot(kind='hist', bins=10, title='Histogram of Earthquake Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.show()

flood_data = disaster_data[disaster_data['type'] == 'flood']

if not flood_data.empty:
    plt.figure(figsize=(10, 5))
    flood_data['level'].value_counts().plot(kind='bar', title='Flood Levels')
    plt.xlabel('Flood Level')
    plt.ylabel('Frequency')
    plt.show()

plt.figure(figsize=(10, 5))
disaster_counts.plot(kind='bar', title='Disaster Counts')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='alarm', data=disaster_data[disaster_data['type'] == 'fire'])
plt.title('Count of Alarm Types in Fire Incidents')
plt.xlabel('Alarm Types')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=disaster_data[disaster_data['type'] == 'fire'], x='involved', bins=20, kde=True)
plt.title('Distribution of Involved Counts in Fire Incidents')
plt.xlabel('Number of Involved')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
yearly_incidents.plot(kind='bar', title='Yearly Disaster Incidents')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.show()

summary_report = f"""
Disaster Data Summary:
{summary_statistics}

Disaster Counts:
{disaster_counts}

Forecasted number of incidents for the next 12 months:
{forecast}

Is Paranaque City prone to disasters?
Threshold: {threshold} incidents per year
Incidents per year: {incidents_per_year}
Prone to disasters: {'Yes' if is_prone_to_disasters else 'No'}

Year with the most disasters:
Year: {year_with_most_disasters}
Number of incidents: {most_disasters_count}
"""

print(summary_report)
