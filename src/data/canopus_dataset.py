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


atom_decoder = ['Se', 'Si', 'S', 'N', 'I', 'B', 'Br', 'Cl', 'F', 'C', 'O', 'P']

class CanopusDataset(InMemoryDataset):
    types = {'Se': 0, 'Si': 1, 'S': 2, 'N': 3, 'I': 4, 'B': 5, 'Br': 6, 'Cl': 7, 'F': 8, 'C': 9, 'O': 10, 'P': 11}

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
        return ['canopus_all_train.smiles', 'canopus_all_train.smiles', 'canopus_all_val.smiles', 'canopus_all_val.smiles', 'canopus_all_test.smiles', 'canopus_all_test.smiles']
    @property
    def split_file_name(self):
        return ['canopus_all_train.smiles', 'canopus_all_train.smiles', 'canopus_all_val.smiles', 'canopus_all_val.smiles', 'canopus_all_test.smiles', 'canopus_all_test.smiles']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['canopus_tr.pt', 'canopus_val.pt', 'canopus_test.pt'] #, 'canopus_all_test_scaf.pt']

    def download(self):
        """
        MOSES install takes care of downloading for us (or should, at least)
        """
        pass
    

    def process(self):
        preprocess = self.preprocess
        RDLogger.DisableLog('rdApp.*')
        ms_dict = pickle.load(open('./data/canopus/raw/canopus.pkl', 'rb'))
        if self.stage == '': # For Predictive Approach
            sca_list = pickle.load(open('./data/canopus/raw/ranks_total_1727337092398.pkl', 'rb'))
            mol_dict = pickle.load(open('./data/canopus/raw/smiles_dict.pkl', 'rb'))
            sca_dict = {}
            for ele in sca_list:
                sca_dict[ele[1]] =mol_dict.get(ele[4][0], None)
        else:
            sca_dict = None
        data = pd.DataFrame(ms_dict)
        
        smiles_list = list(data[data['source'] == self.stage]['smiles']) 
        key_list = list(data[data['source'] == self.stage]['identifier'])
        ms_list = list(data[data['source'] == self.stage]['ms'])

        data_list = []
        smiles_kept = []
        for i, smiles in enumerate(tqdm(smiles_list)):
            data = create_scaffold_graph(smiles, atom_decoder, i, ms_list[i], sca_dict, key_list[i])
            if data == None:
                continue

            
            if not preprocess:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                continue
            
            else:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
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
        print(len(data_list))
        if preprocess:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, 'new_' + self.stage + '.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

        print(data_list[:10])
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
    
class CanopusDataModule(MolecularDataModule):
    DATASET_CLASS = CanopusDataset
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
         loss = loss
         self.validation_step_outputs.append(loss)
         return loss
     
    def prepare_data(self):
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.data_root)

        datasets = {
                    'train': CanopusDataset(stage='train', root=root_path, preprocess=False),
                    'val': CanopusDataset(stage='val', root=root_path, preprocess=False),        
                    'test':CanopusDataset(stage='test', root=root_path, preprocess=False)}
                    
        self.dataloaders = {}
        for split, dataset in datasets.items():
            self.dataloaders[split] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=(self.shuffle and split == 'train'),
            )

        self.train_smiles = datasets['train'].r_smiles
    
    
    
class Canopusinfos(AbstractDatasetInfos):
    max_n_dummy_nodes = 10
    def __init__(self, datamodule, recompute_statistics=False, meta=None):
        self.name = 'Canopus'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {i: Atom(atom).GetMass() for i, atom in enumerate(self.atom_decoder)}
        self.valencies = [1, 3, 4, 3, 2, 1, 4, 1, 2, 4]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 663.0910000000002
        self.max_n_dummy_nodes = 10

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.4679e-05,
        9.4679e-04, 1.7042e-03, 3.7872e-03, 5.7754e-03, 7.5743e-03, 8.7105e-03,
        1.0509e-02, 1.4107e-02, 1.9409e-02, 1.8746e-02, 2.1871e-02, 2.2818e-02,
        2.6037e-02, 2.8025e-02, 3.7114e-02, 3.5410e-02, 3.8629e-02, 4.2700e-02,
        3.2948e-02, 3.5978e-02, 3.9671e-02, 3.8818e-02, 3.6262e-02, 4.3742e-02,
        3.5505e-02, 2.9066e-02, 3.7588e-02, 3.3990e-02, 2.7836e-02, 2.8498e-02,
        1.9883e-02, 1.7042e-02, 1.8084e-02, 1.2498e-02, 1.1077e-02, 1.3539e-02,
        1.3634e-02, 1.2782e-02, 8.7105e-03, 5.8701e-03, 6.7222e-03, 8.2371e-03,
        7.8584e-03, 6.6275e-03, 5.6807e-03, 6.3435e-03, 5.4914e-03, 7.9530e-03,
        5.6807e-03, 5.3967e-03, 3.6925e-03, 2.9351e-03, 3.6925e-03, 2.9351e-03,
        2.1776e-03, 2.1776e-03, 2.8404e-03, 3.1244e-03, 3.4084e-03, 1.9883e-03,
        2.8404e-03, 2.7457e-03, 1.0415e-03, 4.7340e-04, 7.5743e-04, 1.8936e-04])
        # print(self.n_nodes)
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.node_types = torch.tensor([4.0299e-06, 4.0299e-06, 3.5342e-03, 4.2189e-02, 1.4508e-04, 8.0597e-06,
        1.7328e-04, 3.0667e-03, 2.2970e-03, 7.4245e-01, 2.0496e-01, 1.1646e-03])
        self.edge_types = torch.tensor([9.3503e-01, 5.2671e-02, 1.2267e-02, 2.8492e-05, 0.0000e+00])
        self.valency_distribution = None
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0000, 0.1791, 0.3040, 0.3021, 0.2122, 0.0012, 0.0014])

        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

def create_scaffold_graph(smiles, atom_decoder, i, ms, sca_dict=None, key_list=[], use_scaffold=True):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveAllHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    pyg_graph = molecule_to_pyg_graph(mol, atom_decoder, smiles, ms, key=key_list)
    if sca_dict != None:
        p_mol = Chem.MolFromSmiles(sca_dict[key_list[i]])
        pyg_graph = molecule_to_pyg_graph(p_mol, atom_decoder, smiles, ms, key_list[i])
        scaffold = GetScaffoldForMol(p_mol)
        if scaffold == None:
            return None
        scaffold_nodes = align_scaffold_to_molecule(p_mol, scaffold)
        edge_start, edge_end = pyg_graph.edge_index
        edge_mask = torch.tensor([start.item() in scaffold_nodes and end.item() in scaffold_nodes for start, end in zip(edge_start, edge_end)])

        scaffold_edge_index = pyg_graph.edge_index[:, edge_mask]
        scaffold_edge_attr = pyg_graph.edge_attr[edge_mask]
    else:
        scaffold = GetScaffoldForMol(mol)
        if scaffold == None:
            return None
        scaffold_nodes = align_scaffold_to_molecule(mol, scaffold)
        # Create a mask for edges that only connect nodes within the scaffold
        edge_start, edge_end = pyg_graph.edge_index
        edge_mask = torch.tensor([start.item() in scaffold_nodes and end.item() in scaffold_nodes for start, end in zip(edge_start, edge_end)])

        # Apply mask to edge_index to keep only edges in the scaffold
        scaffold_edge_index = pyg_graph.edge_index[:, edge_mask]
        scaffold_edge_attr = pyg_graph.edge_attr[edge_mask]

        # if use_scaffold:
        #     scaffold_edge_index = pyg_graph.edge_index[:, edge_mask]
        #     scaffold_edge_attr = pyg_graph.edge_attr[edge_mask]
        # else:
        #     scaffold_edge_index = torch.zeros_like(pyg_graph.edge_index[:, edge_mask])
        #     scaffold_edge_attr = torch.zeros_like(pyg_graph.edge_attr[edge_mask])

    y = torch.zeros(size=(1, 0), dtype=torch.float)
    data = Data(
        x=pyg_graph.x, edge_index=pyg_graph.edge_index, edge_attr=pyg_graph.edge_attr, y=pyg_graph.y, s=pyg_graph.s, idx=i,
        p_x=pyg_graph.x, p_edge_index=scaffold_edge_index, p_edge_attr=scaffold_edge_attr,
        r_smiles=Chem.MolToSmiles(mol), p_smiles=Chem.MolToSmiles(scaffold), s_mask=pyg_graph.s_mask,
    )
    return data

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def molecule_to_pyg_graph(mol, atom_decoder, smiles, ms=[], energy=0, formula='', key=None, molatom=None):    
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    Chem.Kekulize(mol, clearAromaticFlags=True)
    N = mol.GetNumAtoms()    # try:
    arr = ms[:50]
    rows_to_add = 50 - arr.shape[0]
    padding = ((0, rows_to_add), (0, 0))
    ms = np.pad(arr, pad_width=padding, mode='constant', constant_values=-1000)
    # except:
    #     return None
    ms_mask = ms!= -1000
    
    type_idx = []
    if molatom != None:
        atoms = molatom.GetAtoms()
    else:
        atoms = mol.GetAtoms()
        
    explicit_valence = torch.zeros(N, dtype=torch.long)
    for aid, atom in enumerate(atoms):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'H':
            continue
        type_idx.append(atom_encoder[atom.GetSymbol()])
        explicit_valence[aid] = atom.GetExplicitValence()
        
    row, col, edge_type = [], [], []
    if molatom != None:
        molatom_symbols = [atom.GetSymbol() for atom in molatom.GetAtoms()]
        mol_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # Step 2: Create a mapping from mol's atom indices to molatom's atom indices based on the atom symbols
        atom_map = {}
        for i, symbol in enumerate(mol_symbols):
            # Find the index of the same atom in molatom
            try:
                if symbol not in molatom_symbols or symbol == '*':
                    continue
                j = molatom_symbols.index(symbol)
                molatom_symbols[j] = '_'
                atom_map[i] = j
            except ValueError:
                print(Chem.MolToSmiles(mol), Chem.MolToSmiles(molatom))
                raise Exception(f"Atom {symbol} in mol not found in molatom")
    else:
        atom_map = {atom.GetIdx(): atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}
        
    # Loop through each bond in mol
    for bond in mol.GetBonds():
        # Reorder the bond indices according to the atom_map from molatom
        start = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        
        # Add both directions of the bond (undirected graph)
        row += [start, end]
        col += [end, start]
        
        # Bond types: encoded as edge attributes
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
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=s, explicit_valence=explicit_valence, s_mask = ms_mask)

def align_scaffold_to_molecule(mol, scaffold):
    # Find subgraph of molecule that matches scaffold
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
    
    # Add extra atoms to the scaffold
    for atom_symbol in extra_atoms:
        new_atom = Chem.Atom(atom_symbol)
        editable_scaffold.AddAtom(new_atom)

    # Finalize the scaffold molecule with added atoms
    new_scaffold = editable_scaffold.GetMol()
    
    return new_scaffold


def get_fully_connect(edge_index, edge_attr):
    num_nodes = edge_index.max().item() + 1

    # Create a dense adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    # Create a fully connected adjacency matrix
    fully_connected_adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)

    # Generate all possible edges (edge_index) for the fully connected graph
    fully_connected_edge_index = dense_to_sparse(fully_connected_adj)[0]

    # Create a mapping of existing edges to their attributes
    existing_edges_set = set(map(tuple, edge_index.t().tolist()))
    edge_attr_dict = {tuple(edge): attr for edge, attr in zip(edge_index.t().tolist(), edge_attr)}

    # Initialize new edge_attr with zeros (or any default value you prefer)
    new_edge_attr = []
    attr_dim = edge_attr.size(1)
    for edge in fully_connected_edge_index.t().tolist():
        edge_tuple = tuple(edge)
        new_edge_attr.append(torch.ones(attr_dim)) # Default value for new edges

    new_edge_attr = torch.stack(new_edge_attr)

    # Convert new_edge_attr to appropriate shape and type
    new_edge_attr = new_edge_attr.float()
    
    return fully_connected_edge_index, new_edge_attr


def remove_wildcard_hydrogens(mol):
    """
    Removes atoms like [*H] or [*] from the RDKit molecule.
    """
    editable_mol = Chem.EditableMol(mol)  # Create an editable version of the molecule
    atoms_to_remove = []
    
    # Loop through atoms in the molecule
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*' or (atom.GetSymbol() == 'H'):
            atoms_to_remove.append(atom.GetIdx())
    
    # Remove atoms in reverse order to avoid indexing issues
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(atom_idx)
    
    return editable_mol.GetMol()

def expand_formula(formula):
    # Regular expression to match elements and their optional counts
    element_pattern = r"([A-Z][a-z]*)(\d*)"
    
    # List to hold expanded elements
    expanded_elements = []
    elements_pos = []
    # Find all elements and their counts in the formula
    matches = re.findall(element_pattern, formula)
    
    for (element, count) in matches:
        # Convert count to integer, defaulting to 1 if not specified
        if element == 'H':
            continue
        count = int(count) if count else 1
        # Append the element repeated by its count to the list
        for i in range(count):
            elements_pos.append(i)
            expanded_elements.append(element)
    
    # Join all expanded elements into a single string
    return expanded_elements, elements_pos


if __name__ == '__main__':
    datamodule = CanopusDataModule(
        data_root='/cluster/tufts/liulab/yiwan01/MADGEN/data/canopus/',
        batch_size=64,
        num_workers=0,
        shuffle=False,
        extra_nodes=False,
        swap=False,
        evaluation=False,
        )
