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
         
    
class GAT_GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT_GCN, self).__init__()


        
        self.dropout=Dropout(0.2)
        self.conv1 = GENConv(32, 128,t=1.0, learn_t=True, num_layers=2, norm='layer') # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        #self.conv2 = GENConv(hidden_channels, hidden_channels,
                       # t=1.0, learn_t=True, num_layers=2, norm='layer')
        self.l=Linear(32,2)
        self.conv2 = GCNConv(128, 64)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(128, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, 1)
        '''
        self.layers = torch.nn.ModuleList()
        for i in range(1, 7):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                        t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, 1)
        '''    
        
        #self.prot_li=nn.Linear(300,128)
        self.pool3 = nn.MaxPool1d(kernel_size=1)
        self.conv_x = nn.Conv1d(in_channels=32, out_channels=63, kernel_size=1)
    
        self.l1= Linear(32, 1)
        self.fc11 = nn.Linear(hidden_channels*2, 128)
        self.fc12 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)
    

    def forward(self, proteins, edge_index, edge_attr, x, batch):
            #proteins, edge_index, edge_attr, x, batch= data.target, data.edge_index, data.edge_attr , data.x, data.batch
            edge_encoder = nn.Linear(edge_attr.shape[1], x.shape[1], dtype=torch.float32).to(device)
           
            edge_attr = edge_encoder(edge_attr.float())

            x =  self.conv1(x ,edge_index)
            x = self.bn1(x)
            
            
            x = gmp(x, batch)
            x = F.relu(self.fc1(x))
            x = self.bn3(x)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.fc3(x)
           
            drugs = x
            '''
            x = self.layers[0].conv(x, edge_index, edge_attr)

            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)

            x = self.layers[0].act(self.layers[0].norm(x))
            lin1= nn.Linear(x.shape[0], 32).to('cuda')
            x=lin1(x.T)
            drugs=self.lin(x.T)
            '''
           
                
            conv_xt = self.conv_x(torch.unsqueeze(proteins,dim=-1)).squeeze(dim=-1).relu()
            conv_xt1= self.pool3(conv_xt)
            
         
            '''drugs= self.drop3(self.conv_x(torch.unsqueeze(drugs.T,dim=-1)).squeeze(dim=-1)).relu()
            drugs=self.pool3(drugs)
            drugs=self.flat3(drugs)
            '''
            xc = torch.cat(( drugs,conv_xt1), -1)
           
            xc = F.dropout(self.fc11(xc), p=0.1, training=self.training)
            #lin2= Linear(drugs.shape[0]+1,1)

            xc = self.fc12(xc).relu()
           
           
            #xc = self.lin2(xc.T)
            out = self.out(xc)
          
            return out


'''
class GAT_GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT_GCN, self).__init__()            
        
        

        self.conv_xt1 = nn.Conv1d(in_channels=300, out_channels=hidden_channels, kernel_size=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv_x = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1)
        self.drop3 = Dropout(0.2)#(self.conv_x)
        self.pool3 = nn.MaxPool1d(kernel_size=2)#(self.drop3)
        self.flat3 = nn.Flatten()#(self.pool3)
        self.l1= Linear(128, 1)
        self.dropout= Dropout(0.2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)
    

    def forward(self, data):
            proteins, drugs, batch= data.target, data.drugs, data.batch
            nn.init.xavier_uniform_(proteins)
            nn.init.xavier_uniform_(drugs)
            conv_xt = self.conv_xt1(torch.unsqueeze(proteins,dim=-1)).squeeze(dim=-1).relu()
            conv_xt= self.pool3(self.drop3(conv_xt))
            conv_xt=self.flat3 (conv_xt)
            
            #xt= self.l1(p)
            #print(xt.shape)
            drugs = gmp(drugs, batch)
            drugs= self.drop3(self.conv_x(torch.unsqueeze(drugs,dim=-1)).squeeze(dim=-1)).relu()
            drugs=self.pool3(drugs)
            drugs=self.flat3(drugs)

            xc = torch.cat((drugs, conv_xt), -1)
            xc = self.fc1(xc).relu()
            #lin2= Linear(drugs.shape[0]+1,1)

            xc = self.dropout(xc)
            xc = self.fc2(xc).relu()
           
            xc = self.dropout(xc)
            #xc = self.lin2(xc.T)
            out = self.out(xc)
          
            return out


        self.n_output = n_output
       # self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10 )
       # self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(1560, 1560)
        self.fc_g2 = torch.nn.Linear(1560, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.conv_xt_1 = GCNConv(num_features_xt*10,1)
        self.fc1_xt = nn.Linear(300 , 128)
        self.n_output = n_output
        self.l1=torch.nn.Linear(78,2)
        self.l2=torch.nn.Linear(214,78)
        self.conv1 = SAGEConv(78,2)
        self.conv2 = GCNConv(78,2)
        self.conv3 = SAGEConv(2,124)
        self.fc_g1 = torch.nn.Linear(2, 8)
        self.fc_g2 = torch.nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512 )
        self.out = nn.Linear(512 , self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
         
        x, edge_index, batch = data.x, data.edge_index.view(torch.float32), data.batch
        target = data.target
       
        x, edge_index, edge_attr, batch=  data.x, data.edge_index, data.edge_attr, data.batch
        print( x.shape, edge_index.shape, batch.shape, edge_attr.shape, target.shape)

        x=self.l1(x)
        print(x.shape)
        try:
            x1 = self.conv2(  edge_attr,edge_index.T   )
        except :
            x1=torch.cat((edge_attr,edge_index.T) ,dim=1)
        x1 = self.relu(x1)
       
        #try:
        x1 = self.conv2(x.view(torch.int64), x1) #torch.tensor(x.T).type(torch.int64) 
        #except:
        #    lin1=torch.nn.Linear(x1.shape[0],x.shape[0])
        #    x1= lin1(x1.T)
        #    #lin2=torch.nn.Linear(x.shape[1],124)
        #    x1=torch.cat((x,x1.T),dim=1)
            #x1=self.conv3(x,x1.T)
            
        x1 = self.relu(x1)
        
        lin1=torch.nn.Linear(x1.shape[1],1)
        x=lin1(x1)
        x=self.relu(x)
        
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
       
        x = self.fc_g2(x)

        #embedded_xt = target
        #conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        
        xt = self.fc1_xt(target.float())

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        
    
        out = self.out(xc)
    
        return out
'''