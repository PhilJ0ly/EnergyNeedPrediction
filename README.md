# Quebec Energy Consumption Prediction

The goal of this project is to use past energy usage data supplied by [_Hydro-Québec_](https://www.hydroquebec.com/documents-data/open-data/electricity-demand-quebec/) to estimate future needs in energy.
Energy consumption data since 2019, coupled with weather data from [_Climate Canada_](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html)
will be used as features for a support vector regression, deep neural network, Long short-term memory network, gated recurrent units, and Convolutional neural network to predict energy demand.
This project attempts to predict something complex like energy usage, using easier values to estimate like temperature.

## Web Integration

The performance of the models are displayed on a [web app](), contrasting the models' predictions with 2024 data on energy usage
offered [_Hydro-Québec_](https://www.hydroquebec.com/documents-data/open-data/electricity-demand-quebec/). The user can alternate between the different models and compare the predicted and actual
charts of energy usage.

## Results

### Time Series Approach

| Performance Metrics | LSTM    | GRU     | RNN     | CNN     | CNN-GRU |
| ------------------- | ------- | ------- | ------- | ------- | ------- |
| MAE                 | 135.138 | 175.595 | 224.093 | 151.693 | 156.006 |
| $R^2$               | 0.998   | 0.997   | 0.995   | 0.997   | 0.997   |

### Static Approach

| Performance Metrics | SVR     | DNN     |
| ------------------- | ------- | ------- |
| MAE                 | 568.183 | 672.941 |
| $R^2$               | 0.982   | 0.971   |

<!-- ## Models

-   Support Vector Regression (SVR)
    -   Reasonning: SVR is a good choice of model for this problem as it combines the power of linear regression and the kernel representation of SVM. This permits the features' projection to a infinite-dimensional space , without the associated time complexity. This allows the model to capture the data's complexity and variance.
    -   Architecture and Hyper-Parameters:
-   Deep Neural Network (DNN)
    -   Reasonning: A DNN can do a similar job to SVR. However, the advantage of DNN is that it removes the choice of kernel from the equation. A DNN, through multiple layers, will, in essence, find the optimal "kernel" of the size of its ultimate hidden layer. This allows the model to pick up on complex trends like the ones in energy usage.
    -   Architecture and Hyper-Parameters:
-   Sequencial Convolutional Neural Network (CNN)
    -   Reasonning: While CNNs are traditionally applied to image processing tasks, they can be adapted to analyze one-dimensional sequences effectively. By treating time series data as a spatial signal, CNNs can learn to extract features across different time scales, making them suitable to predict energy consumption
    -   Architecture and Hyper-Parameters:
-   Recurrent Neural Network (RNN)
    -   Reasonning: RNNs excel at capturing temporal dependencies in time series data. With their recurrent connections, RNNs maintain memory of past inputs, allowing them to effectively model sequences and predict future values. This makes them a natural choice for tasks such as energy usage prediction.
    -   Architecture and Hyper-Parameters:
-   Long Short-Term Memory Network (LSTM)
    -   Reasonning: LSTM is a good choice of model as it can capture complex patterns in time-series data. The main advantage of using a LSTM over a RNN is that unlike LSTMs, RNNs can struggle with long-term dependencies due to the vanishing and exploding gradient problems, where gradients either diminish or grow exponentially as they propagate through time.
    -   Architecture and Hyper-Parameters:
-   Gated Recurrent Units (GRU)
    -   Reasonning: GRU is a good choice of model as it offers some of the advantages of a LSTM without its inherent size. The sheer amount of parameters in a LSTM can significantly slow down training and predictions. A GRU is a great lightweight alternative to a LSTM.
    -   Architecture and Hyper-Parameters:
-   CNN-GRU
    -   Reasonning: The combination of convolutional layers and GRU offer the benefits of a CNN in capturing local patterns and hierarchies within the data with the capabilities of GRUs for picking up long-term behaviour in the data. This combination makes it perfect for energy usage data, which fluctuates on hourly to yearly time steps. Here a GRU was used over a LSTM for a more compact model.
    -   Architecture and Hyper-Parameters: -->

## Data Processing

Weather data was taken from four weather stations: Montreal, Quebec, Sherbrooke, and Gatineau. The weather value was then calculated as a weighted average of these hourly values based on the regions' population.
The data was then concatenated and posted to [_Kaggle_](https://www.kaggle.com/datasets/philippejoly/quebec-electrical-power-output-with-temperature).
