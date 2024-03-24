import torch
import torch.nn as nn
import numpy as np

class HYBRID(nn.Module):
    def __init__(self, num_features, output_size, hidden_sizes, num_layers, window_size, k_size=[3,3], pad=[1,1]):
        super(HYBRID, self).__init__()
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.hidden_size = hidden_sizes[2]

        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=hidden_sizes[0], kernel_size=k_size[0], padding=pad[0])
        self.conv2 = nn.Conv1d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=k_size[1], padding=pad[1])
        
        self.relu = nn.ReLU()
        self.gru = nn.GRU(num_features, hidden_sizes[2], num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc2 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc3 = nn.Linear(hidden_sizes[4], output_size)
        
        self.means = np.array([21588.544112986925, 7.351155719146408, 2.4791096271216016e-05, 9.252163087475858e-05, -0.0017768329249686448, 0.0036191039144778535, 0.0772578073503864, -0.026802696954576655])
        self.stds = np.array([5337.969209207278, 12.237716191415544, 0.7070774519619157, 0.7071361027073885, 0.7073399847760873, 0.7068620027194684, 0.6957247570033704, 0.7136385003150916])

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size) 
        
        out = self.conv1(X)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out, _ = self.gru(X, h0)
        out = out[:,-1,:]
        out = self.relu(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
    
    def scaleX(self, X):
        for i in range(self.input_size):
            X[:,:,i] = (X[:,:,i]-self.means[i])/self.stds[i]
        return X
    
    def unscaley(self,y):
        y = y*self.stds[0]+self.means[0]
        return y

    def test_score(self, X_test, scaled=False):
        with torch.no_grad():
            if(not scaled):
                X_test = self.scaleX(X_test)
            outputs = self.forward(torch.from_numpy(X_test).to(torch.float32)).detach().numpy()
            y_pred = self.unscaley(outputs)

            return y_pred