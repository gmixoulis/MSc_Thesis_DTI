import biovec
import json
from collections import OrderedDict
import pickle5 as pickle
import numpy as np


#extracts protein features, from sequnnces to numpy vectors using biovec tool
def protein_features_extraction():

 with open('protein_sequences.txt') as f: #file with whole fasta type (>sp till the end of sequence as values of the dictionary and Prot ID as key)
    data = f.read()
    protein_dict = json.loads(data, object_pairs_hook=OrderedDict)
    
   

    with open('proteins.fasta',"w") as pr:   
        for i in protein_dict.values():    
            pr.write(i)
        pr.close()

    pv = biovec.models.ProtVec("proteins.fasta", corpus_fname="proteins.txt", n=3) #  3-gram 

    # The n-gram "QAT" should be trained in advance
    pv["QAT"]

    # convert whole amino acid sequence into vector
  
    
    with open('protein_seq_2.txt') as f: # file with a dictionary, keys are proteins id and values are proteins sequences
        data = f.read()
    protein_dict = json.loads(data, object_pairs_hook=OrderedDict)
    vs=[]
    ks=[]
    for i,j in protein_dict.items():
        ks.append(i)
        vs.append(np.ndarray.flatten((np.array(pv.to_vecs(protein_dict[i]))))) # from sequnces to vectors, flattened, thus the shape will be (,300)
   
    fe_vec=dict(zip(ks,vs)) #create final dict with keys prot id and values the vectorized features
    
    with open('protein_features.pickle', 'wb') as handle:
     pickle.dump(fe_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    print("Protein Feature Extraction Completed!")