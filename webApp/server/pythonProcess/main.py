import sys
import json
import pandas as pd
import torch
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from modelClasses.dnn import neuralNet
from modelClasses.scnn import SCNN
from modelClasses.lstm import LSTM
from modelClasses.hyb import HYBRID
from modelClasses.gru import GRU
from modelClasses.rnn import RNN
from sklearn.svm import SVR


with open(sys.argv[1], 'r') as f:
    power = json.loads(f.read())
with open(sys.argv[2], 'r') as f:
    temp = json.loads(f.read())

# extract power values
details = power['details']
dates = []
power = []

# Extracting date and demandeTotal values
for detail in details:
    if "demandeTotal" in detail.get("valeurs", {}):
        dates.append(detail["date"])
        power.append(detail["valeurs"]["demandeTotal"])

dfPow = pd.DataFrame({'Date/Time': dates, 'Average Power Output (MW)': power})
dfPow['Date/Time'] = pd.to_datetime(dfPow['Date/Time'])

# treat temps
pop = {"MTL": 0.7607414955721273, "QC": 0.15175753932695674,
       "GAT": 0.056212652146995064, "SHE": 0.03128831295392086}

dfTemp = pd.DataFrame.from_dict(temp, orient='index')

# Calculate weighted average
weighted_avg = (pop["MTL"]*dfTemp['MTL'] + pop["QC"]*dfTemp['QC'] +
                pop["SHE"]*dfTemp['SHE'] + pop["GAT"]*dfTemp['GAT'])

# Create DataFrame with date and weighted average columns
result_dfTemp = pd.DataFrame(
    {'Date/Time': dfTemp.index, 'Temp (°C)': weighted_avg})
result_dfTemp['Date/Time'] = pd.to_datetime(result_dfTemp['Date/Time'])

# concat to main data
data = pd.merge(result_dfTemp, dfPow, on='Date/Time', how='outer')

# add other
data['Year'] = data['Date/Time'].dt.year
data['Month'] = data['Date/Time'].dt.month
data['Day'] = data['Date/Time'].dt.day
data['Hour'] = data['Date/Time'].dt.hour
data['Day of Week'] = data['Date/Time'].dt.strftime("%w")
data['Population'] = 8450000

# data.to_json('weighted_average.json', orient='records')
order = ["Date/Time",	"Temp (°C)", "Average Power Output (MW)",
         "Year", "Month", "Day", "Hour", "Day of Week", "Population"]
data = data[order]
data.dropna(inplace=True)


# prepare data for models
# sequential
sDf = data.copy()
day = 60*60*24
year = 365.2425*day
week = 7*day
sDf['Seconds'] = sDf['Date/Time'].map(pd.Timestamp.timestamp)
sDf['Day sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / day))
sDf['Day cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / day))
sDf['week sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / week))
sDf['week cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / week))
sDf['Year sin'] = np.sin(sDf['Seconds'] * (2 * np.pi / year))
sDf['Year cos'] = np.cos(sDf['Seconds'] * (2 * np.pi / year))

sDf.drop(columns=['Year', 'Month', 'Day', 'Hour',
         'Day of Week', 'Population', 'Seconds'], inplace=True)
sDf.set_index('Date/Time', inplace=True)

column_order = ['Average Power Output (MW)'] + [
    col for col in sDf.columns if col != 'Average Power Output (MW)']
sDf = sDf[column_order]

# sequential formatting
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


Xs, ys = df_to_Xy(sDf, 24)

trained_path = './server/pythonProcess/trainedModels'
# non sequential
with open(f'{trained_path}/Xscaler.pkl', 'rb') as f:
    svrX_scaler = pickle.load(f)
with open(f'{trained_path}/yscaler.pkl', 'rb') as f:
    svry_scaler = pickle.load(f)

nsDf = data.copy()
nsDf.dropna(inplace=True)
nsDf.drop(columns=['Date/Time', 'Year'], inplace=True)
nsDf = nsDf.astype(float)

Xns = nsDf.drop(columns=["Average Power Output (MW)"]).values
yns = nsDf["Average Power Output (MW)"].values


# Load models
with open(f'{trained_path}/svr_bayes_rbf', 'rb') as f:
    svr = pickle.load(f)

dnn = neuralNet(n_features=Xns.shape[1], n_outs=1, hidden_size=10)
dnn.load_state_dict(torch.load(
    f"{trained_path}/dnn_5x5hid_relu__24-03-17_13-35.pth"))
dnn.eval()


gru = GRU(input_size=Xs.shape[2], output_size=1, num_layers=5, hidden_size=150)
gru.load_state_dict(torch.load(
    f"{trained_path}/gru_5x150hid_24wind_24-03-17_16-13.pth"))
gru.eval()

lstm = LSTM(input_size=Xs.shape[2], output_size=1,
            num_layers=5, hidden_size=150)
lstm.load_state_dict(torch.load(
    f"{trained_path}/lstm_5x150hid_24win_24-03-18_00-47.pth"))
lstm.eval()

rnn = RNN(input_size=Xs.shape[2], output_size=1, num_layers=4, hidden_size=150)
rnn.load_state_dict(torch.load(
    f"{trained_path}/rnn_4x150hid_wind24_24-03-17_13-58.pth"))
rnn.eval()

scnn = SCNN(num_features=Xs.shape[2], output_size=1, hidden_sizes=[
            64, 128, 32, 8], window_size=24, k_size=[3, 3], pad=[1, 1])
scnn.load_state_dict(torch.load(
    f"{trained_path}/scnn_c64-c128-l32-l8hid_24-03-17_15-52.pth"))
scnn.eval()

hyb = HYBRID(num_features=Xs.shape[2], output_size=1, hidden_sizes=[
             32, 64, 64, 32, 8], num_layers=3, window_size=24, k_size=[3, 3], pad=[1, 1])
hyb.load_state_dict(torch.load(
    f"{trained_path}/Hybrid_c64-c128-lstm3x150-l32-l8_24-03-17_18-48.pth"))
hyb.eval()

# normalization for svr
X_svr = svrX_scaler.transform(np.copy(Xns))

# Get Predictions
y_svr = svr.predict(X_svr)
y_svr = svry_scaler.inverse_transform(
    y_svr.reshape(len(yns), 1)).reshape(y_svr.shape[0])

y_dnn = dnn.test_score(X_test=Xns, scaled=False)
y_gru = gru.test_score(X_test=Xs, scaled=False)
y_lstm = lstm.test_score(X_test=Xs, scaled=False)
y_rnn = rnn.test_score(X_test=Xs, scaled=False)
y_scnn = scnn.test_score(X_test=Xs, scaled=False)

# add prediction to data as columns
data.sort_values(by='Date/Time', inplace=True)

y_dnn = y_dnn.reshape(y_dnn.shape[0])
y_gru = y_gru.reshape(y_gru.shape[0])
y_lstm = y_lstm.reshape(y_lstm.shape[0])
y_rnn = y_rnn.reshape(y_rnn.shape[0])
y_scnn = y_scnn.reshape(y_scnn.shape[0])

preds = [y_gru, y_lstm, y_rnn, y_scnn]
names = ["GRU", "LSTM", "RNN", "SCNN"]
for i in range(len(names)):
    data[names[i]] = np.pad(
        preds[i], (len(yns)-len(preds[i]), 0), mode='constant')
data["SVR"] = y_svr
data["DNN"] = y_dnn

# get scores
scores = []
for i in range(len(names)):
    scores.append([mean_squared_error(ys, preds[i]), r2_score(
        ys, preds[i]), mean_absolute_error(ys, preds[i])])
scores.append([mean_squared_error(yns, y_svr), r2_score(
    yns, y_svr), mean_absolute_error(yns, y_svr)])
scores.append([mean_squared_error(yns, y_svr), r2_score(
    yns, y_svr), mean_absolute_error(yns, y_svr)])

# return data as json
json_data = data.to_json(orient='records')
json_data = json.loads(json_data)
new_data = {
    "data": json_data,
    "scores": scores
}
new_json_data = json.dumps(new_data)

# write to out
with open('./data/out.json', 'w') as f:
    json.dump(new_data, f)
# print(new_json_data)
