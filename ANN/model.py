import torch
import torch.nn.functional as F

HIDDEN_LAYER1=256
HIDDEN_LAYER2=128
HIDDEN_LAYER3=64
HIDDEN_LAYER4=40
# class FullConnect(torch.nn.Module):
#     def __init__(self):
#         super(FullConnect,self).__init__()
#         self.linear1 = torch.nn.Linear(39,HIDDEN_LAYER1,dtype=torch.float64)
#         self.linear2 = torch.nn.Linear(HIDDEN_LAYER1,HIDDEN_LAYER1, dtype=torch.float64)
#         self.linear3 = torch.nn.Linear(HIDDEN_LAYER1,HIDDEN_LAYER2, dtype=torch.float64)
#         self.linear4 = torch.nn.Linear(HIDDEN_LAYER2,HIDDEN_LAYER2, dtype=torch.float64)
#         self.linear5 = torch.nn.Linear(HIDDEN_LAYER2,HIDDEN_LAYER3, dtype=torch.float64)
#         self.linear6 = torch.nn.Linear(HIDDEN_LAYER3,HIDDEN_LAYER3, dtype=torch.float64)
#         self.linear7 = torch.nn.Linear(HIDDEN_LAYER3,HIDDEN_LAYER4, dtype=torch.float64)
#         self.linear8 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
#         self.linear9 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
#         self.linear10 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
#         self.linear11 = torch.nn.Linear(HIDDEN_LAYER4,2,dtype=torch.float64)
#         self.relu = torch.nn.ReLU()
        
#     def forward(self,x):
#         x = self.relu(self.linear1(x))
#         x = self.relu(self.linear2(x))
#         x = F.dropout(x)
#         x = self.relu(self.linear3(x))
#         x = self.relu(self.linear4(x))
#         x = F.dropout(x)
#         x = self.relu(self.linear5(x))
#         x = self.relu(self.linear6(x))
#         x = F.dropout(x)
#         x = self.relu(self.linear7(x))
#         x = self.relu(self.linear8(x))
#         x = F.dropout(x)
#         x = self.relu(self.linear9(x))
#         x = self.relu(self.linear10(x))
#         x = F.dropout(x)
#         # x = self.sigmoid(self.linear5(x))
#         # x = self.sigmoid(self.linear6(x))
#         # x = self.sigmoid(self.linear7(x))
#         # x = self.sigmoid(self.linear8(x))
#         # x = self.sigmoid(self.linear9(x))
#         #x = F.dropout(x)
#         x = F.log_softmax(self.linear11(x),dim=-1)
#         return x


class FullConnect(torch.nn.Module):
    def __init__(self):
        super(FullConnect,self).__init__()
        self.linear1 = torch.nn.Linear(39,64,dtype=torch.float64)
        self.linear2 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(64,128, dtype=torch.float64)
        self.linear4 = torch.nn.Linear(128,128, dtype=torch.float64)
        self.linear5 = torch.nn.Linear(128,128, dtype=torch.float64)
        self.linear6 = torch.nn.Linear(128,64, dtype=torch.float64)
        self.linear7 = torch.nn.Linear(64,64, dtype=torch.float64)
        self.linear8 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
        # self.linear9 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
        # self.linear10 = torch.nn.Linear(HIDDEN_LAYER4,HIDDEN_LAYER4, dtype=torch.float64)
        self.linear11 = torch.nn.Linear(64,2,dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.linear1(x))
        # x = F.dropout(x)
        x = self.relu(self.linear2(x))
        x = F.dropout(x)
        x = self.relu(self.linear3(x))
        # x = F.dropout(x)
        x = self.relu(self.linear4(x))
        x = F.dropout(x)
        x = self.relu(self.linear5(x))
        #x = F.dropout(x)
        x = self.relu(self.linear6(x))
        x = F.dropout(x)
        x = self.relu(self.linear7(x))
        #x = F.dropout(x)
        # x = self.relu(self.linear6(x))
        # x = self.relu(self.linear7(x))
        # x = self.relu(self.linear8(x))
        # x = self.relu(self.linear9(x))
        # x = self.relu(self.linear10(x))
        # x = self.sigmoid(self.linear5(x))
        # x = self.sigmoid(self.linear6(x))
        # x = self.sigmoid(self.linear7(x))
        # x = self.sigmoid(self.linear8(x))
        # x = self.sigmoid(self.linear9(x))
        x = F.log_softmax(self.linear11(x),dim=-1)
        return x


class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        HIDDEN_LAYER = 64
        self.input = torch.nn.Linear(33,HIDDEN_LAYER,dtype=torch.float64)
        self.hidden = torch.nn.Linear(HIDDEN_LAYER,HIDDEN_LAYER, dtype=torch.float64)
        self.output = torch.nn.Linear(HIDDEN_LAYER,2, dtype=torch.float64)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=-1)
        return x 