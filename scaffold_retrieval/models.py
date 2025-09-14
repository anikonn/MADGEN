# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:18:01 2023

@author: apurv
"""
import torch
import torch.nn as nn
import dgl
from dgllife.model import GCN, GAT

class MolEnc(nn.Module):
    def __init__(self, params, in_dim):
        super(MolEnc, self).__init__()
        dropout = [params['gnn_dropout'] for _ in range(len(params['gnn_channels']))]
        batchnorm = [True for _ in range(len(params['gnn_channels']))]
        gnn_map = {
            "gcn": GCN(in_dim, params['gnn_channels'], batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, params['gnn_channels'], params['attn_heads'])
        }
        self.GNN = gnn_map[params['gnn_type']]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
        self.fc1_graph = nn.Linear(params['gnn_channels'][len(params['gnn_channels']) - 1], params['gnn_hidden_dim'] * 2)
        self.fc2_graph = nn.Linear(params['gnn_hidden_dim'] * 2, params['final_embedding_dim'])
        self.W_out1 = nn.Linear(params['final_embedding_dim'], params['gnn_hidden_dim'] * 2)
        self.W_out2 = nn.Linear(params['gnn_hidden_dim'] * 2, params['gnn_hidden_dim'])
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(params['final_embedding_dim'] * 2)

    def forward(self, g1, f1):
        f = self.GNN(g1, f1)
        h = self.pool(g1, f)
        h1 = self.relu(self.fc1_graph(h))
        h1 = self.dropout(h1)
        h1 = self.fc2_graph(h1)
        h1 = self.dropout(h1)
        #h1 = self.relu(h1)
        
        return h1

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
    
class SpecEncMLP_BIN(nn.Module):
    def __init__(self, params, bin_size):
        super(SpecEncMLP_BIN, self).__init__()

        self.dropout = nn.Dropout(params['fc_dropout'])
        self.mz_fc1 = nn.Linear(bin_size, params['final_embedding_dim'] * 2)
        self.mz_fc2 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'] * 2)
        self.mz_fc3 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.relu = nn.ReLU()
        self.aggr_method = params['aggregator']
        
    def aggr(self, mzvec):
        if self.aggr_method == 'sum':
            aggr_ret = torch.sum(mzvec, axis=1)
        elif self.aggr_method == 'mean':
            aggr_ret = mzvec.sum(axis=1) / ~pad.sum(axis=-1).unsqueeze(-1)
        elif self.aggr_method == 'maxpool':
            input_mask_expanded = torch.where(pad==True, -1e-9, 0.).unsqueeze(-1).expand(mzvec.size()).float()
            aggr_ret = torch.max(mzvec-input_mask_expanded, 1)[0] # Set padding tokens to large negative value
            
        return aggr_ret
    
    def forward(self, mzi_b, int_b, pad, lengths):
                
       h1 = self.mz_fc1(mzi_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       mz_vec = self.dropout(mz_vec)
              
       #mz_vec = self.aggr(mz_vec)
       
       return mz_vec

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

class SpecEncMLP_SIN(nn.Module):
    def __init__(self, params):
        super(SpecEncMLP_SIN, self).__init__()

        self.dropout = nn.Dropout(params['fc_dropout'])
        self.mz_fc1 = nn.Linear(params['sinus_embed_dim'], params['sinus_embed_dim'] * 2)
        self.mz_fc2 = nn.Linear(params['sinus_embed_dim'] * 2, params['sinus_embed_dim'])
        self.mz_fc3 = nn.Linear(params['sinus_embed_dim'], params['sinus_embed_dim'])
        self.mzint_fc1 = nn.Linear(params['sinus_embed_dim'] + 1, params['sinus_embed_dim'] * 2)
        self.mzint_fc2 = nn.Linear(params['sinus_embed_dim'] * 2, params['final_embedding_dim'])
        self.relu = nn.ReLU()
        self.aggr_method = params['aggregator']
        
    def aggr(self, mzvec, pad):
        new_pad = pad.unsqueeze(2)
        mzvec = ~new_pad * mzvec
        if self.aggr_method == 'sum':
            aggr_ret = torch.sum(mzvec, axis=1)
        elif self.aggr_method == 'mean':
            aggr_ret = mzvec.sum(axis=1) / ~pad.sum(axis=-1).unsqueeze(-1)
        elif self.aggr_method == 'maxpool':
            input_mask_expanded = torch.where(pad==True, -1e-9, 0.).unsqueeze(-1).expand(mzvec.size()).float()
            aggr_ret = torch.max(mzvec-input_mask_expanded, 1)[0] # Set padding tokens to large negative value
            
        return aggr_ret
    
    def forward(self, mz_b, int_b, pad, lengths):
                
       h1 = self.mz_fc1(mz_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       #mz_vec = self.dropout(h1)
       
       mzint_vec = torch.cat([mz_vec, int_b], axis=2)
       mzint_vec = self.mzint_fc1(mzint_vec)
       mzint_vec = self.relu(mzint_vec)
       mzint_vec = self.dropout(mzint_vec)
       mzint_vec = self.mzint_fc2(mzint_vec)
       mzint_vec = self.dropout(mzint_vec)
       #mzint_vec = self.relu(mzint_vec)
       #mzint_vec = self.dropout(mzint_vec)
       
       mzint_vec = self.aggr(mzint_vec, pad)
       
       return mzint_vec

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

class SpecEncTFM(nn.Module):
    def __init__(self, params):
        super(SpecEncTFM, self).__init__()

        self.dropout = nn.Dropout(params['fc_dropout'])
        self.tfm_dropout = params['tfm_dropout']
        self.d_model = params['tfm_dim']
        self.tfm_nhead = params['tfm_nhead']
        self.num_tfm_layers = params['num_tfm_layers']
        self.dim_feedforward = params['dim_feedforward']
        self.mz_fc1 = nn.Linear(params['sinus_embed_dim'], params['sinus_embed_dim'] * 2)
        self.mz_fc2 = nn.Linear(params['sinus_embed_dim'] * 2, params['sinus_embed_dim'])
        self.mz_fc3 = nn.Linear(params['sinus_embed_dim'], params['sinus_embed_dim'])
        self.mzint_resize = nn.Linear(params['sinus_embed_dim'] + 1, params['tfm_dim'])
        self.mzint_fc1 = nn.Linear(params['tfm_dim'], params['final_embedding_dim'] * 2)
        self.mzint_fc2 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.tfm_enc = torch.nn.Transformer(d_model = self.d_model, nhead=self.tfm_nhead, num_encoder_layers=self.num_tfm_layers, dim_feedforward=self.dim_feedforward, batch_first=True).encoder
        #self.tfm_enc = torch.nn.Transformer(d_model = self.d_model, nhead=self.nhead, num_encoder_layers=self.num_layers, dim_feedforward=self.dim_feedforward, dropout=self.tfm_dropout).encoder
        self.relu = nn.ReLU()
        self.aggr_method = params['aggregator']
        
    def aggr(self, mzvec, pad, lengths):
        # new_pad = pad.unsqueeze(2)
        # mzvec = ~new_pad * mzvec
        # if self.aggr_method == 'sum':
        #     aggr_ret = torch.sum(mzvec, axis=1)
        # elif self.aggr_method == 'mean':
        #     aggr_ret = mzvec.sum(axis=1) / ~pad.sum(axis=-1).unsqueeze(-1)
        # elif self.aggr_method == 'maxpool':
        #     input_mask_expanded = torch.where(pad==True, -1e-9, 0.).unsqueeze(-1).expand(mzvec.size()).float()
        #     aggr_ret = torch.max(mzvec-input_mask_expanded, 1)[0] # Set padding tokens to large negative value
        aggr_ret = [mzvec[idx,i+1,:] for idx, i in enumerate(lengths)]  
        aggr_ret = torch.stack(aggr_ret)
        
        return aggr_ret
    
    def forward(self, mz_b, int_b, pad, lengths):
                
       h1 = self.mz_fc1(mz_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       mz_vec = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       
       mzint_vec = torch.cat([mz_vec, int_b], axis=2)
       mzint_vec = self.mzint_resize(mzint_vec)
       mzint_vec = self.tfm_enc(mzint_vec, src_key_padding_mask=pad)
       mzint_vec = self.mzint_fc1(mzint_vec)
       mzint_vec = self.relu(mzint_vec)
       mzint_vec = self.dropout(mzint_vec)
       mzint_vec = self.mzint_fc2(mzint_vec)
              
       mzint_vec = self.aggr(mzint_vec, pad, lengths)
       
       return mzint_vec

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

class FP_MLP(nn.Module):
    def __init__(self, params):
        super(FP_MLP, self).__init__()
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.fp_fc1 = nn.Linear(params['final_embedding_dim'], params['final_embedding_dim'] * 2)
        self.fp_fc2 = nn.Linear(params['final_embedding_dim']*2, params['fp_len'])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, spec_b):
        
       h1 = self.fp_fc1(spec_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.fp_fc2(h1)
       h1 = self.sigmoid(h1)
       
       return h1

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

class INTER_MLP(nn.Module):
    def __init__(self, params):
        super(INTER_MLP, self).__init__()
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.fp_fc1 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.fp_fc2 = nn.Linear(params['final_embedding_dim'], params['final_embedding_dim'] // 2)
        self.fp_fc3 = nn.Linear(params['final_embedding_dim'] // 2, params['final_embedding_dim'] // 4)
        self.fp_fc4 = nn.Linear(params['final_embedding_dim'] // 4, params['final_embedding_dim'] // 8)
        self.fp_fc5 = nn.Linear(params['final_embedding_dim'] // 8, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, mol, spec):
        
        y_cat = torch.cat((mol, spec), 1) 
            
        h = self.fp_fc1(y_cat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        z_interaction = self.fp_fc5(h)
        #z_interaction = self.sigmoid(h)
       
        return z_interaction

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
        
class INTER_MLP2(nn.Module):
    def __init__(self, params):
        super(INTER_MLP2, self).__init__()
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.fp_fc1 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.fp_fc2 = nn.Linear(params['final_embedding_dim'], params['final_embedding_dim'] // 2)
        self.fp_fc3 = nn.Linear(params['final_embedding_dim'] // 2, params['final_embedding_dim'] // 4)
        self.fp_fc4 = nn.Linear(params['final_embedding_dim'] // 4, params['final_embedding_dim'] // 8)
        self.fp_fc5 = nn.Linear(params['final_embedding_dim'] // 8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, mol, spec):
        
        y_cat = torch.cat((mol, spec), 1) 
            
        h = self.fp_fc1(y_cat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fp_fc5(h)
        z_interaction = self.sigmoid(h)
       
        return z_interaction

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
