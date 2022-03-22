import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GENConv, DeepGCNLayer, GCNConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gad
# GCN-CNN based model

# GCN-CNN based model



device=  torch.device('cpu')     
         
    
class GEN_Conv(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GEN_Conv, self).__init__()


        
        self.dropout=Dropout(0.2)
        self.conv1 = GENConv(32, 128,t=1.0, learn_t=True, num_layers=2, norm='layer') # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        #self.conv2 = GENConv(hidden_channels, hidden_channels,
                       # t=1.0, learn_t=True, num_layers=2, norm='layer')
        self.target_encoder=Linear(300,32)
        self.l=Linear(32,2)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(128, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 1)
       
        
        #self.prot_li=nn.Linear(300,128)
        self.pool3 = nn.MaxPool1d(kernel_size=1)
        self.conv_x = nn.Conv1d(in_channels=32, out_channels=63, kernel_size=1)
    
        self.l1= Linear(32, 1)
        self.fc11 = nn.Linear(hidden_channels*2, 128)
        self.fc12 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)
    

    def forward(self, proteins, edge_index, edge_attr, x, batch):
            edge_encoder = nn.Linear(edge_attr.shape[1], x.shape[1], dtype=torch.float32).to(device)
            proteins=self.target_encoder(proteins.to(device)).to(device)
            print(x.shape,proteins.shape)
            edge_attr = edge_encoder(edge_attr.float())
            x = F.dropout(x, p=0.2, training=self.training)
            x =  self.conv1(x ,edge_index ,edge_attr )
            
            x = gmp(x, batch)
            
            x = F.relu(self.fc1(x))
            x = self.bn3(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
           
            drugs = x
           
           
                
            conv_xt = self.conv_x(torch.unsqueeze(proteins,dim=-1)).squeeze(dim=-1).relu()
         
            xc = torch.cat(( drugs,conv_xt), -1)
           
            xc = self.fc11(xc)

            xc = self.fc12(xc).relu()
           
           
            out = self.out(xc)
            print(out.shape)
          
            return out

