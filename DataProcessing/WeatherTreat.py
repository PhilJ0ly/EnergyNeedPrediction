#Philippe Joly
#MAIS202

#This is to convert and treat weather data

#this will take weather data in csv files, extract temperatures across different cities, and do a wieghted average of them (according to population). Then the data will be axtracted to a final csv file

import pandas as pd
import numpy as np

root = 'C:\\Users\\philj\\Downloads\\'

years=[i for i in range(2019,2023)]
months=['{:02d}'.format(num) for num in range(1,13)]

cities = {"Montreal":7025251, "Quebec":7016293, "Gatineau":7032685, "Sherbrooke": 7028123}

pop={"MONTREAL INTL A": 3675219.0, "QUEBEC INTL A":733156.0, "OTTAWA GATINEAU A":271569.0, "SHERBROOKE": 151157.0}
# totPop =0
# for i in pop.keys():
#     totPop += pop[i]
# for i in pop.keys():
#     pop[i] = pop[i]/totPop


def treatOne(city, year, month,df):
    lk = root+f'en_climate_hourly_QC_{cities[city]}_{month}-{year}_P1H.csv'
    return pd.concat([df, pd.read_csv(lk, encoding='utf-8')[df.columns]], axis=0, ignore_index=True)

def wAverage(df):
    dfNew = pd.DataFrame(columns=['Date/Time (UTC)', 'Temp (°C)'])
    dates = df['Date/Time (UTC)'].unique()
    count = 1
    for i in dates:
        if(count%4320==0):
            print(i)
        wAv = 0
        totPop=0
        for j, row in df[df['Date/Time (UTC)']==i].iterrows():
            if(not np.isnan(row["Temp (°C)"])):
                wAv += pop[row["Station Name"]]*row["Temp (°C)"]
                totPop += pop[row["Station Name"]]
        if(totPop!=0):
            wAv /= totPop
        else:
            wAv = np.nan
            print('No Value', i)
        new_row = pd.DataFrame({'Date/Time (UTC)':[i], 'Temp (°C)':[wAv]})
        dfNew = pd.concat([dfNew, new_row], ignore_index=True)
        count+=1
    return dfNew

df = pd.DataFrame(columns=['Station Name', 'Date/Time (UTC)', 'Temp (°C)'])
print("--Treat Cities--")
for i in cities.keys():
    print(f'Treating {i}')
    for j in years:
        for k in months:
            df = treatOne(i,j,k,df)

print(f'\n--Averaging--')
final = wAverage(df)
final.to_csv('Temp_2019-2022.csv', index=False, mode='w')
print('DONE!')