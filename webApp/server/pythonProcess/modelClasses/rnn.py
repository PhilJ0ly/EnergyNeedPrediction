import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.means = np.array([21588.544112986925, 7.351155719146408, 2.4791096271216016e-05, 9.252163087475858e-05, -0.0017768329249686448, 0.0036191039144778535, 0.0772578073503864, -0.026802696954576655])
        self.stds = np.array([5337.969209207278, 12.237716191415544, 0.7070774519619157, 0.7071361027073885, 0.7073399847760873, 0.7068620027194684, 0.6957247570033704, 0.7136385003150916])

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size) 
        
        out, _ = self.rnn(X, h0)
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.output_layer(out)
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