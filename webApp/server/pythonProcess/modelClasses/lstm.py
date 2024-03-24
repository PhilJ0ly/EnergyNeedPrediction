import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, hidden_size//5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self.means = np.array([21811.866719292328, 6.583650100621533, 2.624939605327035e-05, 9.796407974698737e-05, -0.0022889574303666554, 0.005353194656774614, 0.08272821882828887, 0.02669666549504437])
        self.stds = np.array([5394.539371301309, 12.119144920239101, 0.7070757266484535, 0.7071378270878812, 0.7075408353244244, 0.7066484772001314, 0.7105731061630806, 0.6982329057404099])

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        out, _ = self.lstm(X, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def scaleX(self, X):
        for i in range(self.input_size):
            X[:, :, i] = (X[:, :, i]-self.means[i])/self.stds[i]
        return X

    def unscaley(self, y):
        y = y*self.stds[0]+self.means[0]
        return y

    def test_score(self, X_test, scaled=False):
        with torch.no_grad():
            if(not scaled):
                X_test = self.scaleX(X_test)
            outputs = self.forward(torch.from_numpy(X_test).to(torch.float32)).detach().numpy()
            y_pred = self.unscaley(outputs)

            return y_pred