import sys
import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle

from modelClasses.dnn import neuralNet
from modelClasses.scnn import SCNN
from modelClasses.lstm import LSTM
from modelClasses.hyb import HYBRID
from modelClasses.gru import GRU
from modelClasses.rnn import RNN
from sklearn.svm import SVR


power = json.loads(sys.argv[1])
tempMTL = json.loads(sys.argv[2])
tempQC = json.loads(sys.argv[3])
tempSH = json.loads(sys.argv[4])
tempGAT = json.loads(sys.argv[5])

# extract power values
details = power['details']
dates = [detail['date'] for detail in details]
power = [detail['valeurs']['demandeTotal'] for detail in details]

data = pd.DataFrame({'date': dates, 'demandeTotal': power})
data['date'] = pd.to_datetime(data['date'])

# treat temps
pop = {"MONTREAL INTL A": 3675219.0, "QUEBEC INTL A": 733156.0,
       "OTTAWA GATINEAU A": 271569.0, "SHERBROOKE": 151157.0}


def wAverage(df):
    dfNew = pd.DataFrame(columns=['Date/Time (UTC)', 'Temp (°C)'])
    dates = df['Date/Time (UTC)'].unique()
    count = 1
    for i in dates:
        wAv = 0
        totPop = 0
        for j, row in df[df['Date/Time (UTC)'] == i].iterrows():
            if (not np.isnan(row["Temp (°C)"])):
                wAv += pop[row["Station Name"]]*row["Temp (°C)"]
                totPop += pop[row["Station Name"]]
        if (totPop != 0):
            wAv /= totPop
        else:
            wAv = np.nan
        new_row = pd.DataFrame({'Date/Time (UTC)': [i], 'Temp (°C)': [wAv]})
        dfNew = pd.concat([dfNew, new_row], ignore_index=True)
        count += 1
    return dfNew

# get all temp values from json to dfTemp
# apply average
# concat to main data
# df = pd.merge(dfTemp, dfPow, on='Date/Time (UTC)', how='outer')

# other fields
# df['Year'] = df['Date/Time (UTC)'].dt.year
# df['Month'] = df['Date/Time (UTC)'].dt.month
# df['Day'] = df['Date/Time (UTC)'].dt.day
# df['Hour'] = df['Date/Time (UTC)'].dt.hour
# df['Day of Week'] = df['Date/Time (UTC)'].dt.strftime("%w")

# prepare data for models
# sequential


def df_to_Xy(df, window_size):
    dfArr = df.to_numpy()
    X = []
    y = []
    for i in range(len(dfArr)-window_size):
        row = [r for r in dfArr[i:i+window_size]]

        if (np.isnan(row).any() or np.isnan(dfArr[i+window_size][0])):
            continue

        X.append(row)
        label = dfArr[i+window_size][0]
        y.append(label)
    return np.array(X), np.array(y)


sDf = data.copy()
day = 60*60*24
year = 365.2425*day
week = 7*day
sDf['Seconds'] = sDf['Date/Time (UTC)'].map(pd.Timestamp.timestamp)
sDf['Day sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / day))
sDf['Day cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / day))
sDf['week sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / week))
sDf['week cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / week))
sDf['Year sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / year))
sDf['Year cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / year))

sDf.drop(columns=['Year', 'Month', 'Day', 'Hour',
         'Day of Week', 'Population', 'Seconds'], inplace=True)
sDf.set_index('Date/Time (UTC)', inplace=True)

column_order = ['Average Power Output (MW)'] + [
    col for col in sDf.columns if col != 'Average Power Output (MW)']
sDf = sDf[column_order]

Xs, ys = df_to_Xy(sDf, 24)

# non sequential
with open('Xscaler.pkl', 'rb') as f:
    svrX_scaler = pickle.load(f)
with open('yscaler.pkl', 'rb') as f:
    svry_scaler = pickle.load(f)

nsDf = data.copy()
nsDf.dropna(inplace=True)
nsDf.drop(columns=['Date/Time (UTC)', 'Year'], inplace=True)
nsDf = nsDf.astype(float)

Xns = nsDf.drop(columns=["Average Power Output (MW)"]).values
yns = nsDf["Average Power Output (MW)"].values


# Load models
dnn = neuralNet(n_features=Xns.shape[1], n_outs=1, hidden_sizes=10)

gru = GRU(input_size=Xs.shape[2], output_size=1, num_layers=5, hidden_size=150)
lstm = LSTM(input_size=Xs.shape[2], output_size=1,
            num_layers=5, hidden_size=150)
rnn = RNN(input_size=Xs.shape[2], output_size=1, num_layers=4, hidden_size=150)
scnn = SCNN(num_features=Xs.shape[2], output_size=1, hidden_sizes=[
            64, 128, 32, 8], window_size=24, k_size=[3, 3], pad=[1, 1])
hyb = HYBRID(num_features=Xs.shape[2], output_size=1, hidden_sizes=[
             32, 64, 64, 32, 8], num_layers=3, window_size=24, k_size=[3, 3], pad=[1, 1])


# sequential formatting

# normalization (do not forget to make copys)

# test

# add prediction to data as columns

# return data as smtg
