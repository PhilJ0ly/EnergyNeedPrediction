import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class neuralNet(nn.Module):
    def __init__(self, n_features, n_outs, hidden_size):
        super(neuralNet, self).__init__()

        self.l1 = nn.Linear(n_features, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.l7 = nn.Linear(hidden_size, n_outs)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        return out

    def getScale(self):
        with open('./trainedModels/Xscaler.pkl', 'rb') as f:
            self.x_scaler = pickle.load(f)
        with open('./trainedModels/yscaler.pkl', 'rb') as f:
            self.y_scaler = pickle.load(f)

    def scaleX(self, X):
        return self.x_scaler.transform(X)

    def unscaley(self, y):
        return self.y_scaler.inverse_transform(y).reshape(y.shape[0])

    def test_score(self, X_test, y_test):
        with torch.no_grad():
            X_test = self.scaleX(X_test)
            outputs = self.forward(torch.from_numpy(
                X_test).to(torch.float32)).detach().numpy()
            y_pred = self.unscaley(outputs)

            return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), y_pred
