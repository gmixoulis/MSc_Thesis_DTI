
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ogb.utils.features import ( atom_to_feature_vector,bond_to_feature_vector) 
import numpy as np
from rdkit import Chem
import torch.nn.functional as F
from drug_embed import drug_embeddings


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
            data_list.append(data)
     
        print('Graph construction done. Saving to file.')
        return data_list



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
  