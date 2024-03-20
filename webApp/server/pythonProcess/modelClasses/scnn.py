import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

class SCNN(nn.Module):
    def __init__(self, num_features, output_size, hidden_sizes, window_size, k_size=[3,3], pad=[1,1]):
        super(SCNN, self).__init__()
        
        self.num_features = num_features

        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=hidden_sizes[0], kernel_size=k_size[0], padding=pad[0])
        self.conv2 = nn.Conv1d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=k_size[1], padding=pad[1])
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_sizes[1]*self.num_features, hidden_sizes[2])
        self.fc2 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3 = nn.Linear(hidden_sizes[3], output_size)


    def forward(self, X):
        out = self.conv1(X)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

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