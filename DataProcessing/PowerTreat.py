#Philippe Joly
#MAIS202

#This combines the data from hourly power output data from hydro quebec
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

root = os.getenv("DOWNLOAD")

years=[i for i in range(2019,2023)]

df = pd.DataFrame(columns=['Date', 'Moyenne (MW)'])
for i in years:
    print(i)
    df = pd.concat([df, pd.read_excel(root+f'{i}-demande-electricite-quebec.xlsx')])

df=df.rename(columns={'Date':'Date/Time (UTC)'})
df.to_csv('Power_2019-2022.csv', index=False, mode='w')
print('DONE!')