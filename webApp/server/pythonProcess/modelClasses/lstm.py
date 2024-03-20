import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, hidden_size//5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        
        out, _ = self.lstm(X, (h0,c0))
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def getScale(self):
        self.means = []
        self.stds = []
        #load from csv

    def scaleX(self, X):
        for i in range(self.input_size):
            X[:,:,i] = (X[:,:,i]-self.means[i])/self.stds[i]
        return X
    
    def unscaley(self,y):
        y = y*self.stds[0]+self.means[0]
        return y

    def test_score(self, X_test, y_test, scaled=False):
        with torch.no_grad():
            if(not scaled):
                X_test = self.scaleX(X_test)
            outputs = self.forward(torch.from_numpy(X_test).to(torch.float32)).detach().numpy()
            y_pred = self.unscaley(outputs)

            return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), y_pred