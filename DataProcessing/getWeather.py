#Philippe Joly
#MAIS202

#This is to extract hourly weather data from the nevironment canada website

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
# import DateTime
from time import sleep as slp

s = Service('C:\\Users\\philj\\edgedriver_win32\\msedgedriver.exe')
error = []

def getData(station, stationName, year, month,t):
    try:
        url=f'https://climate.weather.gc.ca/climate_data/hourly_data_e.html?hlyRange=2013-02-13%7C2024-02-17&dlyRange=2013-02-14%7C2024-02-17&mlyRange=%7C&StationID={station}&Prov=QC&urlExtension=_e.html&searchType=stnName&optLimit=yearRange&StartYear=2020&EndYear=2024&selRowPerPage=25&Line=0&searchMethod=contains&txtStationName={stationName}&timeframe=1&time=LST&time=UTC&Year={year}&Month={month}&Day=1#'

        driver.get(url)
        slp(3)

        downBut = driver.find_element(By.XPATH, '//input[@value="Download Data"]')
        downBut.click()
        print(f'{stationName}-{year}-{month} Downloaded')
        slp(1)
    except:
        if(t<4):
            getData(station, stationName, year, month, t+1)
        else:
            print('--ERROR--')
            error.append(f'{stationName}-{year}-{month}')

years=[i for i in range(2019,2023)]
months=[i for i in range(1,13)]

cities = {"Montreal": 51157, "Quebec":51457, "Gatineau":53001, "Sherbrooke": 48371}


driver = webdriver.Edge(service=s)
driver.minimize_window()

for k in cities.keys():
    print(f'\n--{k}--')
    for i in years:
        print(f'--{i}--')
        for j in months:
            getData(cities[k],k,i,j,0)
driver.quit()

print(f'\n{len(error)} Errors')
for n in error:
    print(error)
