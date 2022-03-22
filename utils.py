import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

def plot_losses(train_loss,test_loss,st):
    def mean(li): 
        return sum( li)/len(li)
    plt.figure(figsize=(14, 4))
    plt.xlabel('epochs')
    plt.ylabel('loss'+st)
    plt.plot([mean(train_loss[i]) for i in range(len(train_loss))],label="train_loss")
    plt.plot([mean(test_loss[i:i+10]) for i in range(len(test_loss))], label="valid_loss")
    plt.legend()
    plt.savefig("oss.png")


def mae(y,f):
    return abs((y - f))

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def trainss(model,device, train_loader, optimizer,epoch):
        print('Training on')
        epoch=epoch
        device=device
        total_preds=torch.Tensor()
        train_loss=[]
        total_labels=torch.Tensor()
        
        for  i in  tqdm(train_loader, desc='Train data on epoch  {}'.format(epoch)) :
            data = i.to(device)   
            model.train()
            optimizer.zero_grad()
            output = model(data.target, data.edge_index, data.edge_attr, data.x, data.batch)
            loss =  F.mse_loss(output, data.y.view(-1,1))
            loss.backward(retain_graph=True)
            optimizer.step()
            total_preds = torch.cat((total_preds, output.cpu()), -1)
                
            total_labels = torch.cat((total_labels, data.y.cpu()), -1)
                
            train_loss.append(loss)
            

            
        print('***Train epoch: {} tLoss: {:.6f}'.format(epoch, loss.item()))
        return total_labels.flatten(),total_preds.flatten(),train_loss






def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss=[]
  
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output =  model(data.target,data.edge_index, data.edge_attr,data.x,data.batch) 
            loss =  F.mse_loss(output.flatten(), data.y.flatten().to(device))
            total_loss.append(loss)
         
            total_preds = torch.cat((total_preds, output.cpu().flatten()), -1)
          
            total_labels = torch.cat((total_labels, data.y.flatten().cpu()), -1)
   
    MSE = mean_squared_error(total_labels, total_preds)

    print("MSE",MSE)
    total_losss=sum(total_loss)/len(total_loss)
        
    return total_labels.numpy() ,total_preds.numpy() , total_losss
