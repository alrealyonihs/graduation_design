import torch
import torch.nn as nn
import torch.nn.functional as F 

input_size = 39
hidden_size = 128
num_layers = 3
dropout_rate = 0.5

class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=39,    # num of input unit
                            hidden_size=256,  # num of input unit
                            num_layers=3,    # num of hidden layers
                            dropout=0.5,
                            batch_first=True)         # True£º[batch, time_step, input_size] False:[time_step, batch, input_size]
        
        #batch normalizition
        self.batch_norm = nn.BatchNorm1d(num_features=256)

        # fully conncetion layer
        self.lin = nn.Linear(in_features=256,
                              out_features=64)
        self.lin2 = nn.Linear(in_features=64,
                              out_features=64)

        #output layer
        self.output_layer = nn.Linear(in_features=64,    # num of input layer
                                       out_features=2)  # num of output layer
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x, _ = self.lstm(x, None)   
        x = self.batch_norm(x)
        x = self.lin(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.output_layer(x) 
        x = F.log_softmax(x, dim=-1) 
        return x

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # Transformer layer
        self.transformer = nn.TransformerDecoderLayer()
        
        #batch normalizition
        self.batch_norm = nn.BatchNorm1d(num_features=256)

        # fully conncetion layer
        self.lin = nn.Linear(in_features=256,
                              out_features=64)
        #output layer
        self.output_layer = nn.Linear(in_features=64,    # num of input layer
                                       out_features=2)  # num of output layer

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x, _ = self.transformer(x, None)
        x = F.dropout(x,p=0.5)   
        x = self.batch_norm(x)
        x = self.lin(x)
        x = self.output_layer(x) 
        x = F.log_softmax(x, dim=-1) 
        return x


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        #input
        self.input_layer = nn.Linear(input_size, hidden_size)

        #LSTM
        self.lstm_layers = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        #Dropout
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        self.bn_layer = nn.BatchNorm1d(hidden_size)

        #fully conncetion layer
        self.fc_layer = nn.Linear(hidden_size, 64)

        # Softmax output
        self.softmax_layer = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #input layer
        x = self.input_layer(x)

        #LSTM layer
        out, _ = self.lstm_layers(x)

        # Dropout layer
        out = self.dropout_layer(out)

        out = self.bn_layer(out)

        #full connect layer
        out = nn.functional.relu(self.fc_layer(out))

        # Softmax output layer
        out = self.softmax_layer(out)
        out = self.softmax(out)

        return out