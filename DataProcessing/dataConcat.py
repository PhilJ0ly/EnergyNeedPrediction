#Philippe Joly 
#MAIS 202

#This will concatenate the entireity of the data (date, power, temp, population)into one csv file
import pandas as pd
from datetime import datetime as dt

#quarterky population data from Quebec from statistics Canada
pop = [8430363,	8447632,	8483186,	8521542,	8537376,	8550900,	8551095,	8551865,	8550561,	8556015,	8572020,	8603553,	8613999,	8627524,	8672185,	8730868]


dfTemp = pd.read_csv('Temp_2019-2022.csv', encoding='utf-8')
dfTemp['Date/Time (UTC)'] = pd.to_datetime(dfTemp['Date/Time (UTC)'])

dfPow = pd.read_csv('Power_2019-2022.csv', encoding='utf-8')
dfPow['Date/Time (UTC)'] = pd.to_datetime(dfPow['Date/Time (UTC)'])

df = pd.merge(dfTemp, dfPow, on='Date/Time (UTC)', how='outer')

df= df.iloc[:-2]

df['Year'] = df['Date/Time (UTC)'].dt.year
df['Month'] = df['Date/Time (UTC)'].dt.month
df['Day'] = df['Date/Time (UTC)'].dt.day
df['Hour'] = df['Date/Time (UTC)'].dt.hour
df['Day of Week'] = df['Date/Time (UTC)'].dt.strftime("%w")


def get_pop(year, month):
    i = ((year-2019)*12+month-1)//3
    return pop[i]

df['Population']= df.apply(lambda row: get_pop(row['Year'], row['Month']), axis=1)

nan_rows = df[df.isna().any(axis=1)]
print('Working...')

df=df.rename(columns={'Moyenne (MW)':'Average Power Output (MW)'})
df.to_csv('Data_2019-2022.csv', index=False, mode='w')
print('DONE!')
print('total number of rows',df.shape[0])
print('number of rows with missing values',nan_rows.shape[0])
