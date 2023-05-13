import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing



class MineCraft(Dataset):
    def __init__(self,filepath,id):
        #two ways:
        #1.all data load to Memory(structured data)
        #2.define an list ,put each example in one list, put each target in another list
        #xy = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=np.float64)
        xy = pd.read_csv(filepath, delimiter=',', dtype=np.float64)
        xy = xy.drop(columns=["num"],axis=1)
        xy = xy.to_numpy()
        self.len = xy.shape[0]
        #print(self.len)
        userid = id
        positive_xy = []
        nagetive_xy = []
        for i in range(self.len):
            if xy[i,[-1]]==userid: 
                xy[i,[-1]] = 1
                positive_xy.append(xy[i,:])
            else:
                xy[i,[-1]] = 0
                nagetive_xy.append(xy[i,:])

        nagetive_xy = pd.DataFrame(nagetive_xy)
        nagetive_xy = nagetive_xy.sample(len(positive_xy),random_state=np.random.seed(0))
        
        xy = np.concatenate((positive_xy,nagetive_xy.values),axis=0)
        #print(len(xy))
        self.lenth = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.x_data = self.z_score(self.x_data)
        self.y_data = torch.LongTensor(np.transpose(xy[:,-1]))
        
        #print("Data ready...")

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.lenth
    
    def z_score(self, x):
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)
        return x 

class MineCraft_CNN(Dataset):
    def __init__(self,filepath,id):
        #two ways:
        #1.all data load to Memory(structured data)
        #2.define an list ,put each example in one list, put each target in another list
        #xy = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=np.float64)
        xy = pd.read_csv(filepath, delimiter=',', usecols= ["Timestamp","X", "Y","Subject ID"], dtype=np.float64)
        # xy = xy.drop(columns=["num"],axis=1)
        xy = xy.to_numpy()
        self.len = xy.shape[0]
        #print(self.len)
        userid = id
        positive_xy = []
        nagetive_xy = []
        for i in range(self.len):
            if xy[i,[-1]]==userid: 
                xy[i,[-1]] = 1
                positive_xy.append(xy[i,:])
            else:
                xy[i,[-1]] = 0
                nagetive_xy.append(xy[i,:])

        nagetive_xy = pd.DataFrame(nagetive_xy)
        nagetive_xy = nagetive_xy.sample(len(positive_xy),random_state=np.random.seed(0))
        
        xy = np.concatenate((positive_xy,nagetive_xy.values),axis=0)
        #print(len(xy))
        self.lenth = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.x_data = self.z_score(self.x_data)
        self.y_data = torch.LongTensor(np.transpose(xy[:,-1]))
        
        #print("Data ready...")

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.lenth
    
    def z_score(self, x):
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)
        return x 
    

MineCraft_CNN('/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/DL/Minecraft-Mouse-Dynamics-Dataset/masterTrain.csv',0) 

