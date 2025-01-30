import pickle
import os
import pathlib
import re

from typing import Any, Sequence
from collections import Counter
from tqdm import tqdm

import os.path as osp
import numpy as np
import pandas as pd

from rdkit.Chem.rdchem import Atom
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdFMCS

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from src.data import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.data.abstract_dataset import MolecularDataModule, AbstractDatasetInfos

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

atom_decoder = ['N', 'P', 'B', 'I', 'As', 'Se', 'Cl', 'C', 'F', 'S', 'Br', 'O', 'Si']

class MsGymDataset(InMemoryDataset):
    types = {'N': 0, 'P': 1, 'B': 2, 'I': 3, 'As': 4, 'Se': 5, 'Cl': 6, 'C': 7, 'F': 8, 'S': 9, 'Br': 10, 'O': 11, 'Si': 12}

    bonds = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None, preprocess=False):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.remove_h = True
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['msgym_full_train.smiles', 'msgym_full_train.smiles', 'msgym_full_val.smiles', 'msgym_full_val.smiles', 'msgym_full_test.smiles', 'msgym_full_test.smiles']

    @property
    def split_file_name(self):
        return ['msgym_full_train.smiles', 'msgym_full_train.smiles', 'msgym_full_val.smiles', 'msgym_full_val.smiles', 'msgym_full_test.smiles', 'msgym_full_test.smiles']

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['msgym_tr.pt', 'msgym_val.pt', 'msgym_test.pt']

    def download(self):
        pass

    def process(self):
        preprocess = self.preprocess
        RDLogger.DisableLog('rdApp.*')
        ms_dict = pickle.load(open('./data/msgym/raw/msgym.pkl', 'rb'))
        if self.stage is None:
            sca_list = pickle.load(open('./data/msgym/raw/ranks_total_1722912941993.pkl', 'rb'))
            sca_dict = {ele[1]: Chem.MolToSmiles(Chem.MolFromSmarts(ele[4][0])) for ele in sca_list}
        else:
            sca_dict = None
        data = pd.DataFrame(ms_dict)
        
        smiles_list = list(data[data['source'] == self.stage]['smiles']) 
        key_list = list(data[data['source'] == self.stage]['identifier'])
        ms_list = list(data[data['source'] == self.stage]['ms'])
        
        assert len(smiles_list) == len(ms_list)
        data_list = []
        smiles_kept = []
        
        for i, smiles in enumerate(tqdm(smiles_list)):
            data = create_scaffold_graph(smiles, atom_decoder, i, ms_list[i], sca_dict, key_list)
            if data is None:
                continue
            
            if not preprocess:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                continue
            
            else:
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                        smiles = mol2smiles(largest_mol)
                        smiles_kept.append(smiles)
                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
        if preprocess:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, 'new_' + self.stage + '.smiles')
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

class MsGymDataModule(MolecularDataModule):
    DATASET_CLASS = MsGymDataset

    def __init__(self, data_root, batch_size, num_workers, shuffle, extra_nodes=False, evaluation=False, swap=False):
        super().__init__(batch_size, num_workers, shuffle)
        self.extra_nodes = extra_nodes
        self.evaluation = evaluation
        self.swap = swap
        self.data_root = data_root
        self.train_smiles = []
        self.prepare_data()
        self.preprocess = False
        
    def validation_step(self, loss):
         self.validation_step_outputs.append(loss)
         return loss
     
    def prepare_data(self):
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.data_root)

        datasets = {
            'train': MsGymDataset(stage='train', root=root_path, preprocess=False),
            'val': MsGymDataset(stage='val', root=root_path, preprocess=False),
            'test': MsGymDataset(stage='test', root=root_path, preprocess=False)
        }
        self.dataloaders = {}
        for split, dataset in datasets.items():
            self.dataloaders[split] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=(self.shuffle and split == 'train'),
            )

        self.train_smiles = datasets['train'].r_smiles

class MsGyminfos(AbstractDatasetInfos):
    max_n_dummy_nodes = 10

    def __init__(self, datamodule, recompute_statistics=False, meta=None):
        self.name = 'MsGym'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {i: Atom(atom).GetMass() for i, atom in enumerate(self.atom_decoder)}
        self.valencies = [3, 2, 4, 2, 3, 1, 1, 2, 1, 3, 4, 2, 1, 3, 1, 2, 2, 1, 2, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 663.0910000000002
        self.max_n_dummy_nodes = 10

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8820e-04, 8.4241e-04,
        1.8372e-03, 4.2658e-03, 7.4115e-03, 1.5477e-02, 2.1616e-02, 2.5550e-02,
        2.8759e-02, 2.8642e-02, 3.8500e-02, 3.6009e-02, 3.8411e-02, 3.9611e-02,
        4.1296e-02, 3.9853e-02, 5.6460e-02, 5.0482e-02, 5.3735e-02, 4.9174e-02,
        4.5150e-02, 4.2703e-02, 4.5625e-02, 3.5704e-02, 2.8785e-02, 2.8158e-02,
        2.7997e-02, 2.3014e-02, 2.1786e-02, 1.5621e-02, 1.4375e-02, 9.5444e-03,
        8.2180e-03, 5.7087e-03, 4.3107e-03, 6.6676e-03, 5.0634e-03, 6.2912e-03,
        9.4010e-03, 6.1030e-03, 4.5526e-03, 2.2942e-03, 1.7834e-03, 1.3353e-03,
        1.8282e-03, 9.5892e-04, 1.6400e-03, 9.7684e-04, 1.7028e-03, 2.0343e-03,
        1.6131e-03, 1.1650e-03, 5.5564e-04, 9.7684e-04, 1.0665e-03, 7.8864e-04,
        4.3017e-04, 4.9290e-04, 5.6460e-04, 4.3017e-04, 1.3084e-03, 7.5280e-04,
        7.0799e-04, 8.6034e-04, 4.0328e-04, 1.7924e-05, 5.3771e-05, 3.5847e-04])
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.node_types = torch.tensor([8.4720e-02, 1.5525e-03, 8.8791e-06, 6.6860e-04, 6.6594e-06, 3.9956e-06,
        8.1284e-03, 7.2950e-01, 6.3996e-03, 8.7686e-03, 5.4118e-04, 1.5967e-01,
        3.2409e-05])
        self.edge_types = torch.tensor([9.1330e-01, 6.5862e-02, 2.0747e-02, 9.1781e-05, 0.0000e+00])
        self.valency_distribution = None
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0000, 0.1556, 0.2751, 0.3304, 0.2343, 0.0014, 0.0032])

        # if meta is None:
        #     meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        # assert set(meta.keys()) == set(meta_files.keys())
        # for k, v in meta_files.items():
        #     if (k not in meta or meta[k] is None) and os.path.exists(v):
        #         meta[k] = np.loadtxt(v)
        #         setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

def create_scaffold_graph(smiles, atom_decoder, i, ms, sca_dict=None, key_list=[], use_scaffold=True):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveAllHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    pyg_graph = molecule_to_pyg_graph(mol, atom_decoder, smiles, ms)
    
    if sca_dict is not None:
        p_mol = Chem.MolFromSmiles(sca_dict[key_list[i]])
        p_graph = molecule_to_pyg_graph(p_mol, atom_decoder, smiles, ms)
        scaffold = GetScaffoldForMol(p_mol)
        if scaffold is None:
            return None
        scaffold_nodes = align_scaffold_to_molecule(p_mol, scaffold)
        edge_start, edge_end = p_graph.edge_index
        edge_mask = torch.tensor([start.item() in scaffold_nodes and end.item() in scaffold_nodes for start, end in zip(edge_start, edge_end)])

        scaffold_edge_index = p_graph.edge_index[:, edge_mask]
        scaffold_edge_attr = p_graph.edge_attr[edge_mask]
    else:
        scaffold = GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        scaffold_nodes = align_scaffold_to_molecule(mol, scaffold)
        edge_start, edge_end = pyg_graph.edge_index
        edge_mask = torch.tensor([start.item() in scaffold_nodes and end.item() in scaffold_nodes for start, end in zip(edge_start, edge_end)])

        scaffold_edge_index = pyg_graph.edge_index[:, edge_mask]
        scaffold_edge_attr = pyg_graph.edge_attr[edge_mask]

    y = torch.zeros(size=(1, 0), dtype=torch.float)
    data = Data(
        x=pyg_graph.x, edge_index=pyg_graph.edge_index, edge_attr=pyg_graph.edge_attr, y=pyg_graph.y, s=pyg_graph.s, idx=i,
        p_x=pyg_graph.x, p_edge_index=scaffold_edge_index, p_edge_attr=scaffold_edge_attr,
        r_smiles=Chem.MolToSmiles(mol), p_smiles=Chem.MolToSmiles(scaffold), s_mask=pyg_graph.s_mask,
    )
    return data

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def molecule_to_pyg_graph(mol, atom_decoder, smiles, ms_dict={}):    
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    Chem.Kekulize(mol, clearAromaticFlags=True)
    N = mol.GetNumAtoms()
    if len(ms_dict) != 0:
        ms = ms_dict
        arr = ms[:50]
        rows_to_add = 50 - arr.shape[0]
        padding = ((0, rows_to_add), (0, 0))
        ms = np.pad(arr, pad_width=padding, mode='constant', constant_values=-1000)
        ms_mask = ms != -1000
    else: 
        ms_mask = None
    
    type_idx = []

    atoms = mol.GetAtoms()
        
    explicit_valence = torch.zeros(N, dtype=torch.long)
    for aid, atom in enumerate(atoms):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'H':
            continue
        type_idx.append(atom_encoder[atom.GetSymbol()])
        explicit_valence[aid] = atom.GetExplicitValence()
        
    row, col, edge_type = [], [], []
    atom_map = {atom.GetIdx(): atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}
        
    for bond in mol.GetBonds():
        start = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        
        row += [start, end]
        col += [end, start]
        
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]
        
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_encoder)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)

    s = torch.tensor(ms, dtype=torch.float)
    s = s.reshape(-1, 100)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=s, explicit_valence=explicit_valence, s_mask=ms_mask)

def align_scaffold_to_molecule(mol, scaffold):
    mcs = rdFMCS.FindMCS([mol, scaffold])
    smarts = mcs.smartsString
    subgraph = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    return set(subgraph)

def add_missing_atoms(scaffold, atoms_list):
    mol_atoms = [atom.GetSymbol() for atom in scaffold.GetAtoms()]
    
    scaffold_counter = Counter(mol_atoms)
    mol_counter = Counter(atoms_list)
    
    extra_atoms = []
    for atom, count in mol_counter.items():
        if count > scaffold_counter[atom]:
            extra_atoms.extend([atom] * (count - scaffold_counter[atom]))

    editable_scaffold = Chem.EditableMol(scaffold)
    
    for atom_symbol in extra_atoms:
        new_atom = Chem.Atom(atom_symbol)
        editable_scaffold.AddAtom(new_atom)

    new_scaffold = editable_scaffold.GetMol()
    
    return new_scaffold

def get_fully_connect(edge_index, edge_attr):
    num_nodes = edge_index.max().item() + 1

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    fully_connected_adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)

    fully_connected_edge_index = dense_to_sparse(fully_connected_adj)[0]

    existing_edges_set = set(map(tuple, edge_index.t().tolist()))
    edge_attr_dict = {tuple(edge): attr for edge, attr in zip(edge_index.t().tolist(), edge_attr)}

    new_edge_attr = []
    attr_dim = edge_attr.size(1)
    for edge in fully_connected_edge_index.t().tolist():
        edge_tuple = tuple(edge)
        new_edge_attr.append(torch.ones(attr_dim))

    new_edge_attr = torch.stack(new_edge_attr)

    new_edge_attr = new_edge_attr.float()
    
    return fully_connected_edge_index, new_edge_attr

def remove_wildcard_hydrogens(mol):
    editable_mol = Chem.EditableMol(mol)
    atoms_to_remove = []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*' or (atom.GetSymbol() == 'H'):
            atoms_to_remove.append(atom.GetIdx())
    
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(atom_idx)
    
    return editable_mol.GetMol()

def expand_formula(formula):
    element_pattern = r"([A-Z][a-z]*)(\d*)"
    
    expanded_elements = []
    elements_pos = []
    matches = re.findall(element_pattern, formula)
    
    for (element, count) in matches:
        if element == 'H':
            continue
        count = int(count) if count else 1
        for i in range(count):
            elements_pos.append(i)
            expanded_elements.append(element)
    
    return expanded_elements, elements_pos