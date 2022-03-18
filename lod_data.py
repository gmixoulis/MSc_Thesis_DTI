
import torch
import pandas as pd
import numpy as np
import rdkit as rd
import os
import json
from collections import OrderedDict
#from molvecgen import SmilesVectorizer
import matplotlib.pyplot as plt
from rdkit import Chem
import biovec

import pickle5 as pickle

from torch_geometric.data import Data, DataLoader
import torch

from torch_geometric.nn.inits import constant
import models.gat_gcn1 as GAT1
from math import sqrt
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from sklearn.model_selection import  train_test_split

from torch_geometric.nn import GENConv, DeepGCNLayer

from scipy import stats
from ogb.utils.features import ( atom_to_feature_vector,bond_to_feature_vector) 
torch.cuda.empty_cache()


#loss_fn = nn.BCELoss() #regression
loss_fn= nn.MSELoss() #classification


device=torch.device('cpu')

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
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci



def process( xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []

        data_len = len(xd)
        for i in   tqdm(range(data_len), desc = 'Creating Drug Embeddings') :
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            edge_index, edge_attr , x = smile_graph[smiles].edge_index ,smile_graph[smiles].edge_attr, smile_graph[smiles].x
            data = Data(    
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr.type(torch.int64),
                
                 y=torch.FloatTensor([labels]),
                                )
            data.target = torch.FloatTensor([target])
            # append graph, label and target sequence to data list
            data_list.append(data)
            #torch.save(model,"model_drugs.pth")
     
        print('Graph construction done. Saving to file.')
        #data, slices = self.collate(data_list)
        return data_list


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
    #loss= F.softmax(output, -1)
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
    #device=torch.device('cuda:0')
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(loader.dataset)))
    total_loss=[]
    predicted_scores=[]
    correct_labels=[]
    with torch.no_grad():
        for data in loader:
            
            data = data.to(device)
           
            output =  model(data.target,data.edge_index, data.edge_attr,data.x,data.batch) 
            
            #with torch.autocast('cuda'):
           # loss=F.softmax(output, 1).to('cpu').data.numpy()
            loss =  F.mse_loss(output.flatten(), data.y.flatten().to(device))
            total_loss.append(loss)
            correct_labels.append (data.y.to('cpu').data.numpy())
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            ys = output.to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            #predicted_scores = list(map(lambda x: x[1], ys))
            #predicted_scores.append( ys)
            

            total_preds = torch.cat((total_preds, output.cpu().flatten()), -1)
          
            total_labels = torch.cat((total_labels, data.y.flatten().cpu()), -1)
    from sklearn.metrics import roc_auc_score, precision_score, recall_score,mean_squared_error, auc
    MSE = mean_squared_error(total_labels, total_preds)

    print("Metrics",MSE)
    total_losss=sum(total_loss)/len(total_loss)
        
    return total_labels.numpy() ,total_preds.numpy() , total_losss




TRAIN_BATCH_SIZE = 1673 
TEST_BATCH_SIZE =  550 
LR = 0.001



def drug_embeddings(data):
        class DeeperGCN(torch.nn.Module):
                def __init__(self, hidden_channels, num_layers):
                    super().__init__()

                    self.node_encoder = Linear(data.x.size(-1), hidden_channels ,dtype=torch.float32)
                    self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels, dtype=torch.float32)
                    
                    
                    
                    self.layers = torch.nn.ModuleList()
                    for i in range(1, 4 + 1):
                        conv = GENConv(hidden_channels, hidden_channels,
                                    t=1.0, learn_t=True, num_layers=2, norm='layer')
                        norm = LayerNorm(hidden_channels, elementwise_affine=True)
                        act = ReLU(inplace=True)

                        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.0,
                                            ckpt_grad=i % 3)
                        self.layers.append(layer)

                    self.lin = Linear(hidden_channels, 1)

                def forward(self, x, edge_index, edge_attr):
                    
                    x = self.node_encoder(x)
                    edge_attr = self.edge_encoder(edge_attr.float())
                    x = self.layers[0].conv(x, edge_index, edge_attr)

                    for layer in self.layers[1:]:
                        x = layer(x, edge_index, edge_attr)

                    x = self.layers[0].act(x)
                    lin1= nn.Linear(x.shape[0], 32)
                    x=lin1(x.T)
                    out=self.lin(x.T)
                    return out
        model=DeeperGCN(hidden_channels=32, num_layers=12)
        data.drug = torch.FloatTensor(model(data.x, data.edge_index, data.edge_attr))
        
        
        return data

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    from torch_geometric.graphgym import AtomEncoder, BondEncoder
    atom_encoder = AtomEncoder(emb_dim=32)
    bond_encoder = BondEncoder(emb_dim=32)

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64)
       

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.zeros((1, 2), dtype = np.int64)
        edge_attr = np.zeros((1, num_bond_features), dtype =  np.int64)

    data=Data(
    edge_index = torch.LongTensor(edge_index).transpose(-1,0),
    edge_attr= torch.LongTensor(edge_attr),
    x = torch.LongTensor(x))
    atom_emb=atom_encoder(data)
    
    try:
        edge_emb=bond_encoder(atom_emb)
        drug_embedding=drug_embeddings(edge_emb)
          
        return edge_emb, drug_embedding.drug
    except Exception as e:
            print(e)
            
            atom_emb.edge_attr=F.pad(atom_emb.edge_attr, pad=(32 -  atom_emb.edge_attr.shape[1],  0,0, 0 ), value=0.0)
            atom_emb.edge_index= torch.randint(15,(2,atom_emb.edge_attr.shape[0]),dtype=torch.int64)
            drug_embedding=drug_embeddings(atom_emb)
            return atom_emb, drug_embedding.drug
  



def load_data():

    network_path = 'data1/' #windows
    #network_path = '/home/gm/Downloads/thesis_data-20211116T221419Z-001/thesis_data/' #ubuntu
    '''
    with open('file11.txt') as f:
        data = f.read()
    protein_dict = json.loads(data, object_pairs_hook=OrderedDict)
    
    proteins=[]

    with open('proteins.fasta',"w") as pr:   
        for i in protein_dict.values():    
            pr.write(i)
        pr.close()

    pv = biovec.models.ProtVec("proteins.fasta", corpus_fname="proteins.txt", n=3)

    # The n-gram "QAT" should be trained in advance
    pv["QAT"]

    # convert whole amino acid sequence into vector
  
    
    with open('file.txt') as f:
        data = f.read()
    protein_dict = json.loads(data, object_pairs_hook=OrderedDict)
    vs=[]
    ks=[]
    for i,j in protein_dict.items():
        ks.append(i)
        vs.append(np.ndarray.flatten((np.array(pv.to_vecs(protein_dict[i])))))
    for i in vs:
        print(i.shape)
    fe_vec=dict(zip(ks,vs))
    #print(fe_vec)
    with open('v.pickle', 'wb') as handle:
     pickle.dump(fe_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    '''
    with open('file.txt') as f:
        data_pr = f.read()
    protein_dict = json.loads(data_pr, object_pairs_hook=OrderedDict)
    with open('v.pickle', 'rb') as handle:
        pr = pickle.load(handle)

    with open('drugs_smiles.txt') as f:
        data = f.read()
    drug_dict = json.loads(data, object_pairs_hook=OrderedDict)
  
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_path=network_path

    script_dir = os.path.dirname(drug_path) #<-- absolute dir the script is in
    rel_path = "drug.txt"
    abs_file_path = os.path.join(script_dir, rel_path)

    file1 = open(abs_file_path, 'r')
    drugss=[]
    for i in range(708):
        line = file1.readline()
        if not line:
            break
        drugss.append(line.strip())
    
    file1.close()
    script_dir = os.path.dirname(drug_path) #<-- absolute dir the script is in
    rel_path = "protein.txt"
    abs_file_path = os.path.join(script_dir, rel_path)

    file1 = open(abs_file_path, 'r')
    proteinss=[]
    for i in range(1512):
        line = file1.readline()
        if not line:
            break
        proteinss.append(line.strip())
    
    file1.close()


  
    
    dti_orig = np.loadtxt(network_path + 'mat_drug_protein.txt')
    dr_pr_matrix= pd.DataFrame(dti_orig,index=drugss,columns=proteinss)
    dr_pr_matrix.to_csv("original_dataset.csv")
   

    # Removed DTIs with similar drugs or proteins
    #drug_protein = np.loadtxt(network_path + 'mat_drug_protein_homo_protein_drug.txt')




    print("Load data finished.")


    
    whole_positive_index=[]
    whole_negative_index=[]
    for i in range(np.shape(dti_orig)[0]):
        for j in range(np.shape(dti_orig)[1]):
            if int(dti_orig[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_orig[i][j]) == 0:
                whole_negative_index.append([i, j])


        # pos:neg=1:10
    negative_sample = np.random.choice(np.arange(len(whole_negative_index)), size= 10*len(whole_positive_index), replace=False)

    
    positive_data=np.zeros((len(whole_positive_index),3), dtype=int)
    for index, i in enumerate(whole_positive_index):
        for j in range(2):
            positive_data[index][j] = i[j]
    positive_data[:,2]=1

    negative_data=np.zeros((len(negative_sample),3), dtype=int)
    for index, i in enumerate(negative_sample):
        for j in range(2):
            negative_data[index][j] = whole_negative_index[i][j]
        negative_data[:,2]=0
    dataset1=pd.DataFrame(np.concatenate((positive_data,negative_data)), columns=['Drugs', 'Proteins', 'Interaction'])

    temp={}
    for i in dr_pr_matrix.index:
       
        temp[dr_pr_matrix.index.get_loc(i)]=i
  
    temp_pr={}
    for i in range(1512):
        temp_pr[i]=dr_pr_matrix.columns[i]
    dataset1.Drugs.replace(temp, inplace=True)
    dataset1.Proteins.replace(temp_pr, inplace=True)
    dataset1.Drugs.replace(drug_dict,inplace=True)
    #dataset1.Proteins.replace(protein_dict, inplace=True)
    dataset1=dataset1.sample(frac = 1)
  
    loader={}
    druggs=[]
    for  i in drug_dict.keys():
        t=drug_dict[i]
    
        embs1,drug_emb=smiles2graph(t)
        loader[t]=embs1
        druggs.append(embs1) 
    
    train, test = train_test_split(dataset1, test_size=0.13, random_state=np.random.RandomState(82))
   
    train_df = train.sample(frac=1).reset_index(drop=True)
    


   
    prot_features = []
    
   
    prot_features=[]
    with tqdm(total=len(train_df['Proteins'])) as t:
        for col in train_df['Proteins']:
                
                pr_name=col
                target_features= pr[pr_name]
                prot_features.append(target_features)
                t.update(1)
  
    

    train_drugs, train_prots,  train_Y = np.asarray(list(train_df['Drugs'])),np.asarray(list(prot_features)),np.asarray(list(train_df['Interaction']))
  
    test_df = test.sample(frac=1).reset_index(drop=True)
    prot_features_test = []

    
    for col in test_df['Proteins']:
            
            pr_name=col
            target_features= pr[pr_name]
            prot_features_test.append(target_features)

    test_drugs, test_prots,  test_Y = np.asarray(list(test_df['Drugs'])),np.asarray(list(prot_features_test)),np.asarray(list(test_df['Interaction']))

  
    
    
    train_loader1=process( xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=loader)
    train_loader = DataLoader(train_loader1, batch_size=TRAIN_BATCH_SIZE,  shuffle=True)
    test_loader1=process(  xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=loader)
    test_loader = DataLoader(test_loader1, batch_size=TEST_BATCH_SIZE,  shuffle=True)

    
   
    
    model = GAT1.GEN_Conv(hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    result_file_name = 'result_train.csv'
    train_loss=[]
    test_loss=[]
    count=0

    for epoch in tqdm(range(100), desc= "Training:"):
        features,y, d=trainss(model, device, train_loader, optimizer, epoch+1) # features, d=(trainss(modelss, device, train_loader, optimizer,  epoch+1))
        
        train_loss.append(d)
        G,P, loss = predicting(model, device, test_loader)

        from sklearn.metrics import roc_auc_score,precision_score, mean_squared_error
        print(roc_auc_score(G,P))

        test_loss.append(loss)
        
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        
        print(ret)
        from scipy import integrate
        sorted_index = np.argsort(G)
        fpr_list_sorted =  np.array(G)[sorted_index]
        tpr_list_sorted = np.array(P)[sorted_index]
        auc=integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
        print("auc: ",auc)
        from sklearn.metrics import mean_absolute_error
        
        
        print(" ROC",roc_auc_score(G,P),mean_absolute_error(G,P))
        if count==99:
            
            with open(result_file_name,'w') as f:
                f.write(','.join(map(str,ret)))
                f.write(str(model.state_dict()))
        

        
    torch.save(model.state_dict(),'model1.pth')
    plot_losses(train_loss,test_loss,"train_valid_1")
if __name__  == "__main__":    
    load_data()
