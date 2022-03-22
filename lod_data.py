import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
import models.gat_gcn1 as GAT1
from math import sqrt
import torch.nn as nn
from tqdm import tqdm

from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error as rmse
from scipy.stats import pearsonr, spearmanr
from prot_features import protein_features_extraction
from utils import trainss, predicting, plot_losses
from process_data import process, smiles2graph, load_data

torch.cuda.empty_cache()
#loss_fn = nn.BCELoss() #regression
loss_fn= nn.MSELoss() #classification
device=torch.device('cpu')


TRAIN_BATCH_SIZE = 1673 
TEST_BATCH_SIZE =  550 
LR = 0.001

def run():

    network_path = 'data1/' #windows
    protein_features_extraction()

    proteinss, drugss, pr, drug_dict=load_data(network_path)    
    dti_orig = np.loadtxt(network_path + 'mat_drug_protein.txt')
    dr_pr_matrix= pd.DataFrame(dti_orig,index=drugss,columns=proteinss)
    dr_pr_matrix.to_csv("original_dataset.csv")
    print("Load data finished.")


    #seperate the positive interactions from the negative
    whole_positive_index=[]
    whole_negative_index=[]
    for i in range(np.shape(dti_orig)[0]):
        for j in range(np.shape(dti_orig)[1]):
            if int(dti_orig[i][j]) == 1:
                whole_positive_index.append([i, j]) # append the edges, node drug with node protein
            elif int(dti_orig[i][j]) == 0:
                whole_negative_index.append([i, j])


   # pos:neg=1:10 sampling 
    negative_sample = np.random.choice(np.arange(len(whole_negative_index)), size= 10*len(whole_positive_index), replace=False)

    
    positive_data=np.zeros((len(whole_positive_index),3), dtype=int) #initialize positive data
    for index, i in enumerate(whole_positive_index):
        for j in range(2):
            positive_data[index][j] = i[j]
    positive_data[:,2]=1 # all the positive data/ edges are equal with 1

    negative_data=np.zeros((len(negative_sample),3), dtype=int)
    for index, i in enumerate(negative_sample):
        for j in range(2):
            negative_data[index][j] = whole_negative_index[i][j]
        negative_data[:,2]=0 #  all the negative data/ edges are equal with 0
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
        d=trainss(model, device, train_loader, optimizer, epoch+1) # features, d=(trainss(modelss, device, train_loader, optimizer,  epoch+1))
        
        train_loss.append(d)
        G,P, loss = predicting(model, device, test_loader)

     
        
        print(roc_auc_score(G,P))

        test_loss.append(loss)
        
        ret = [rmse(G,P, squared=False),rmse(G,P, squared=True),pearsonr(G,P),spearmanr(G,P)]
        
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
    run()
