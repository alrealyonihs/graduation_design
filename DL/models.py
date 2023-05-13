import torch
import torch.nn as nn
import torch.nn.functional as F

class FullConnect(torch.nn.Module):
    def __init__(self):
        super(FullConnect,self).__init__()
        self.linear1 = torch.nn.Linear(33,256,dtype=torch.float64)
        self.linear2 = torch.nn.Linear(256,256, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(256,128, dtype=torch.float64)
        self.linear4 = torch.nn.Linear(128,128, dtype=torch.float64)
        self.linear5 = torch.nn.Linear(128,64, dtype=torch.float64)
        self.linear6 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear7 = torch.nn.Linear(64,40, dtype=torch.float64)
        self.linear8 = torch.nn.Linear(40,2, dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.relu(self.linear6(x))
        x = self.relu(self.linear7(x))
        #x = F.log_softmax(self.linear8(x),dim=-1)
        x = self.linear8(x)
        x = nn.functional.softmax(x,dim=1)
        return x



class CZHNet(torch.nn.Module):
    def __init__(self):
        super(CZHNet,self).__init__()
        self.linear1 = torch.nn.Linear(33,32,dtype=torch.float64)
        self.linear2 = torch.nn.Linear(32,64, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear4 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear5 = torch.nn.Linear(64,32, dtype=torch.float64)
        self.linear6 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear7 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear8 = torch.nn.Linear(32,2, dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = F.log_softmax(self.linear8(x),dim=-1)
        return x
    


class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=33,    # num of input unit
                            hidden_size=128,  # num of input unit
                            num_layers=3,    # num of hidden layers
                            dropout=0.5,
                            batch_first=True,dtype=torch.float64)         # True��[batch, time_step, input_size] False:[time_step, batch, input_size]
        
        #batch normalizition
        self.batch_norm = nn.BatchNorm1d(num_features=128,dtype=torch.float64)

        # fully conncetion layer
        self.lin = nn.Linear(in_features=128,
                              out_features=64,dtype=torch.float64)
        # self.lin2 = nn.Linear(in_features=64,
        #                       out_features=64,dtype=torch.float64)

        #output layer
        self.output_layer = nn.Linear(in_features=64,    # num of input layer
                                       out_features=2,dtype=torch.float64)  # num of output layer
        
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
        # x = self.lin2(x)
        # x = self.relu(x)
        x = self.output_layer(x) 
        x = F.log_softmax(x, dim=-1) 
        return x
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(20, 20, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(20, 20, 2)
        self.conv3 = nn.Conv1d(20, 20, 2)

        self.gmp = nn.MaxPool1d(1,1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36, 64)
        self.fc2 = nn.Linear(64,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        x = self.gmp(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x