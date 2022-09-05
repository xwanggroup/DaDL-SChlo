#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json  
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm 
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[2]:


class ProteinModel(nn.Module):
    def __init__(self, num_classes=5, init_weights=False):
        super(ProteinModel, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,10)),
            nn.ReLU(inplace=True),
        ) 
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=400, nhead=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(400, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)   
        x = self.conv(x)   
        x = x.squeeze(3)    
        x = self.encoder_layer(x)  
        x = torch.mean(x, axis=1)  
        x = self.classifier(x)    
        return x


# In[7]:


def load_data():

    data = pd.read_csv("dataset.csv").values

    train, test = train_test_split(data, test_size=0.3)
    train_list=train[1:,5:]
    train_ndarry=np.array(train_list).astype("float64") 
    train_lable = train[1:,:5].astype("float64")  
    a = torch.tensor(train_ndarry) 
    b = torch.tensor(train_lable)
    dataload = TensorDataset(a, b) 
    train_loader = DataLoader(dataset=dataload, batch_size=8, shuffle=True)

    test_list = test[1:,5:]
    test_ndarry = np.array(test_list).astype("float64")
    test_lable = test[1:,:5].astype("float64")
    c = torch.tensor(test_ndarry)
    d = torch.tensor(test_lable)
    dataload = TensorDataset(c, d)
    test_loader = DataLoader(dataset=dataload, batch_size=4, shuffle=True)

    return train_loader,test_loader


# In[8]:


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]

def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]

def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


# In[9]:


def main():
    batch_size=4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    train_loader,val_loader=load_data() 
    net = ProteinModel()
    net.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    trainloss=[]
    valoaa=[]
    valacc=[]
    valpre=[]
    valrec=[]
    valf1=[]
    
    epochs = 150
    save_path = './Model.pth'
    best_acc = 0.0
    train_steps = len(train_loader) 
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar): 
            seq, labels = data 
            seq=seq.view(-1,400,10)          
            seq=seq.float()
            optimizer.zero_grad()
            outputs = net(seq.to(device)) 
            loss = loss_function(outputs.float(), labels.float().to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            
        net.eval() 
        oaa = 0.0
        acc = 0.0 
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        with torch.no_grad(): 
            val_bar = tqdm(val_loader)
            for val_data in val_bar: 
                val_seq, val_labels = val_data
                print(val_seq.shape)
                val_seq = val_seq.view(-1, 400,10)  
                val_seq = val_seq.float()
                outputs = net(val_seq.to(device))
#                 predict_y = torch.max(outputs, dim=1)[1] 
#                 acc += torch.eq(outputs.float(), val_labels.float().to(device)).sum().item()
#                 print(outputs)
#                 outputs = (outputs == outputs.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
#                 outputs= outputs.cpu().numpy()
#                 print(outputs)
                outputs_int = np.zeros_like(outputs)
                outputs_int[outputs > 0.5] = 1
                print(outputs_int)
                
                val_labels = val_labels.cpu().numpy()
                oaa += accuracy_score(val_labels, outputs)            
                acc += Accuracy(val_labels, outputs)
                precision += Precision(val_labels, outputs)
                recall += Recall(val_labels, outputs)
                f1 += F1Measure(val_labels, outputs)
                
#                 print('acc: %.3f oaa: %.3f pre: %.3f recall: %.3f f1: %.3f' %
#                           (acc, oaa, precision, recall, f1))
        print(len(val_loader), len(val_bar))
        val_oaa = oaa / (len(val_loader))
        val_accurate = acc / (len(val_loader))
        val_precision = precision / (len(val_loader))
        val_recall = recall / (len(val_loader))
        val_f1 = f1 / (len(val_loader))
        print('[epoch %d] train_loss: %.3f val_oaa: %.3f  val_accuracy: %.3f val_precision: %.3f val_recall: %.3f val_f1 : %.3f' %
              (epoch + 1, running_loss / train_steps,val_oaa, val_accurate, val_precision, val_recall, val_f1))
        trainloss.append(running_loss / train_steps)
        valoaa.append(val_oaa)
        valacc.append(val_accurate)
        valpre.append(val_precision)
        valrec.append(val_recall)
        valf1.append(val_f1)
        
        if val_accurate > best_acc:  
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
#     print(trainloss)
#     print(valacc)
    print('Finished Training')


# In[10]:


if __name__ == '__main__':
    main()


# In[ ]:




