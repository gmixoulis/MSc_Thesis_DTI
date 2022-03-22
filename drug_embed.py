
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer
import torch
from torch import nn


#create drug embeddings for DTI approach

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