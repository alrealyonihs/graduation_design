import csv
import datetime
import torch
from model import CNN,Multi_Scale_CNN
from torch.utils.data import DataLoader
from data import Balabit
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import auc
from logger.logger import logger_setup

FILE = "/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/CNN/data/MNIST/raw/balabit_39feat_PC_MM_DD_1000.csv"
#FILE = "/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/CNN/data/MNIST/raw/master10Test_Extracted.csv"
LEARNING_RATE=0.01
WEIGHT_DECAY=0.01
BATCH_SIZE=20
EPOCH_NUM=100
TEST_SIZE = 0.33
NUM_ACTION = 1

def get_dataset(user,train=True):
    #1.use MyDataset class,build my dataset
    dataset = Balabit(FILE,id=user)
    test_size = int(len(dataset)*TEST_SIZE)
    train_size = len(dataset)-test_size

    #get train and test dataset
    torch.manual_seed(2023)
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])

    #2.use Dataloader build train_loader
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    if train == True:
        return train_loader
    else:
        return test_loader


def datatrain(epoch_num,user):
    datatest_loader = get_dataset(user,True)
    for epoch in range(epoch_num):
        correct = 0
        for i, data in enumerate(datatest_loader,0):
            #1.prepare data
            inputs,labels = data
            optimizer.zero_grad()

            #2.transform ahead
            output = model(inputs)
            loss = criterion(output,labels)
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
            #print(epoch,i,loss.item())
            #logger.info(epoch,i,loss.item())

            #3.transform back
            loss.backward()

            #4.renew
            optimizer.step()
        #print(f'Train Accuracy:{100*correct/len(datatest_loader.dataset)}%')
        print(loss.item())
        logger.info(f'Epoch:{epoch}/{epoch_num},Train Accuracy:{100*correct/len(datatest_loader.dataset)}%')

def datatest(user,num_action):
    loss = 0
    correct = 0
    label=[]
    prob=[]
    model.eval()
    datatest_loader = get_dataset(user,False)
    with torch.no_grad():
        for data,target in datatest_loader:
                
            output = model(data)
            loss += criterion(output,target)
            pred = output.data.max(1,keepdim=True)[1]
            #print(pred)
            for i in range(len(pred)):
                prob.append(pred[i].item())
                label.append(target[i].item())

            # which one is more simple? maybe chabuduo
            # prob.extend(np.transpose(pred.data).tolist()[0])
            # label.extend(target.data.tolist())
            
            correct += pred.eq(target.data.view_as(pred)).sum()
    loss /= len(datatest_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(datatest_loader.dataset),
        100. * correct / len(datatest_loader.dataset)))
    logger.info('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(datatest_loader.dataset),
        100. * correct / len(datatest_loader.dataset)))
    
    fpr,tpr,threshold = roc_curve(label,prob)
    print(f'fpr:{fpr},tpr:{tpr},threshold:{threshold}')
    auc = roc_auc_score(label, prob)
    print(f'auc:{auc}')
    logger.info(f'fpr:{fpr},tpr:{tpr},threshold:{threshold},auc:{auc}')
    csv_writer.writerow([user, f'{100.*correct / len(datatest_loader.dataset)}%', auc])
    

model = CNN()
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

if __name__ == "__main__":
    ids = [7, 9, 12, 15, 16, 20, 21, 23, 29, 35]
    #ids = range(0,10)

    #log path
    log_filename = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/CNN/log/'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.log'
    logger = logger_setup(log_filename)
    logger.info(f'lr:{LEARNING_RATE},weight_decay:{WEIGHT_DECAY},batch_size:{BATCH_SIZE}')
    logger.info(model)

    #write csv
    csv_filename = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/CNN/csv_record/'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.csv'
    csv_file = open(csv_filename, 'w', encoding='utf-8')

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["user","test_acc", "auc"])

    for i in ids:
        print('This is for user'+str(i))
        logger.info('This is for user'+str(i))
        model.zero_grad()
        datatrain(EPOCH_NUM,i)
        datatest(i,NUM_ACTION)
    csv_file.close()