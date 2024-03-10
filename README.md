# Quebec Energy Consumption Prediction
The goal of this project is to use past energy usage data supplied by [_Hydro-Québec_](https://www.hydroquebec.com/documents-data/open-data/electricity-demand-quebec/) to estimate future needs in energy. 
To predict energy demand, energy consumption data since 2019, coupled with weather data from [_Climate Canada_](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html) 
will be used as features for a support vector regression, deep neural network, Long short-term memory network, and Prophet [_Prophet_](https://doi.org/10.7287/peerj.preprints.3190v2) model. 
This project attempts to predict something complex like energy usage, using easier values to estimate like weather.

## Web Integration
The performance of the models are displayed on a [web app](), contrasting the models' predictions with the real time data on energy usage 
offered [_Hydro-Québec_](https://www.hydroquebec.com/documents-data/open-data/electricity-demand-quebec/). The user can alternate between the different models and compare the predicted and actual 
chart of energy usage.

## Results

### Static Approach
| Performance Metrics | SVR | DNN  | CNN |
| --- | --- | --- | --- |
| $R^2$  | Content Cell  | Content Cell  | Content Cell  | 
| MAE  | Content Cell  | Content Cell  | Content Cell  |
| MSE  | Content Cell  | Content Cell  | Content Cell  | 

### Time Series Approach
| Performance Metrics | LSTM | GRU | _Prophet_ | Sequential CNN | MIX1 |
| --- | --- | --- | --- | --- | --- | 
| $R^2$  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | 
| MAE  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| MSE  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |

## Models 
+ Support Vector Regression (SVR)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework:
+ Deep Neural Network (DNN)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework: 
+ Sequencial Convolutional Neural Network (CNN)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework: 
+ Long Short-Term Memory Network (LSTM)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework:
+ Gated Recurrent Units (GRU)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework:  
+ [_Prophet_](https://doi.org/10.7287/peerj.preprints.3190v2)
  - Reasonning:
  - Architecture and Hyper-Parameters:
  - Framework:

## Data Processing
Weather data was taken from four weather stations: Montreal, Quebec, Sherbrooke, and Gatineau. The weather value was then calculated as a weighted average of these hourly values based on the regions' population.
The data was then concatenated and posted to [_Kaggle_](https://www.kaggle.com/datasets/philippejoly/quebec-electrical-power-output-with-temperature). 
