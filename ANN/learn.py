from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F

train_batch_size=64
test_batch_size=1000
img_size=28
def get_dataloader(train=True):
    assert isinstance(train,bool),"train is a bool"
    dataset = torchvision.datasets.MNIST('/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/ANN/data',train=train,download=True,
                                         transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),]))
    #data loader
    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

class LYLNet(torch.nn.Module):
    def __init__(self):
        super(LYLNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28*1,28)
        self.fc2 = torch.nn.Linear(28,10)
    def forward(self,x):
        x = x.view(-1,28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
         
model = LYLNet()
optimizer = optim.Adam(model.parameters(),lr=0.001)
#criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

train_loss_list = []
train_count_list = []

def train(epoch):
    model.train(True)
    train_dataloader = get_dataloader(True)
    for idx,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss{:.6f}'.format(epoch, idx  * len(data), len(train_dataloader.dataset),100. * idx / len(train_dataloader),loss.item())) 
            train_loss_list.append(loss.item())
            train_count_list.append(idx*train_batch_size+(epoch-1)*len(train_dataloader))

#epoch = 1
#for i in range(epoch):
 #   train(i)

def test():
    test_loss = 0
    correct = 0
    model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data,target in test_dataloader:
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            #get the max location,[batch_size]
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

if __name__ == '__main__':

    test()
    for i in range(1):
        train(i)
        test()