import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


# class Balabit(Dataset):
#     def __init__(self,filepath,id):
#         #two ways:
#         #1.all data load to Memory(structured data)
#         #2.define an list ,put each example in one list, put each target in another list
#         xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
#         self.len = xy.shape[0]
#         userid = id
#         j1 = 0
#         j2 = 0
#         positive_xy = np.zeros(shape=(int(self.len*0.1),40),dtype=np.float32)
#         nagetive_xy = np.zeros(shape=(int(self.len*0.9),40),dtype=np.float32)
#         for i in range(self.len):
#             if xy[i,[-1]]==userid: 
#                 xy[i,[-1]] = 1
#                 positive_xy[j1,:] = xy[i,:]
#                 j1 += 1
#             else:
#                 xy[i,[-1]] = 0
#                 nagetive_xy[j2,:] = xy[i,:]
#                 j2 +=1
#         #print(j1,j2)

#         new_nagetive_xy = np.zeros(shape=(int(self.len*0.1),40),dtype=np.float32)
#         row_nagetive = np.arange(nagetive_xy.shape[0])
#         np.random.shuffle(row_nagetive)
#         new_nagetive_xy = nagetive_xy[row_nagetive[0:int(self.len*0.1)]]
#         #print(new_nagetive_xy)
#         new_xy = np.zeros(shape=(int(self.len*0.2),40),dtype=np.float32)
#         new_xy = np.concatenate((positive_xy,new_nagetive_xy))
#         #print(new_xy)
#         self.lenth = new_xy.shape[0]

#         self.x_data = torch.from_numpy(new_xy[:,0:-1])
        
#         self.y_data = torch.LongTensor(np.transpose(new_xy[:,-1]))
#         #print(self.x_data[:,0:20])
#         #print(self.y_data)
#         #print("Data ready...")

#     def __getitem__(self, index):
#         return self.x_data[index],self.y_data[index]
    
#     def __len__(self):
#         return self.lenth

class Balabit(Dataset):
    def __init__(self,filepath,id):
        #two ways:
        #1.all data load to Memory(structured data)
        #2.define an list ,put each example in one list, put each target in another list
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
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
        nagetive_xy = nagetive_xy.sample(len(positive_xy),random_state=np.random.seed(2023))
        
        xy = np.concatenate((positive_xy,nagetive_xy.values),axis=0)
        #print(xy)
        self.lenth = xy.shape[0]
        #number_test = int(self.lenth*TEST_SIZE)
        #if train==False:
        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.y_data = torch.LongTensor(np.transpose(xy[:,-1]))
        #else:
            # self.x_data = torch.from_numpy(xy[number_test:self.lenth,0:-1])
            # self.y_data = torch.from_numpy(np.transpose(xy[number_test:self.lenth,-1]))
        # self.testlen = xy.shape[0]*TEST_SIZE
        # print(self.decide)
        # print(self.testlen)
        # self.trainlen = xy.shape[0]-self.testlen
        # print(self.trainlen)
        #print(self.x_data[:,0:20])
        #print(self.y_data)
        #print("Data ready...")

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.lenth