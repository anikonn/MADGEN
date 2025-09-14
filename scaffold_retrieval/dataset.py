# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:09:22 2023

@author: apurv
"""
import dgl
import dgllife.utils as chemutils
import torch
from collections import defaultdict, Counter
import rdkit.Chem as Chem
import numpy as np
import pickle
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import random

def atom_bond_type_one_hot(atom):
    bs = atom.GetBonds()
    if len(bs) == 0:
        return [False, False, False, False]
    bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
    # print([i.GetBondTypeAsDouble() for i in bs], bt)
    # print([any(bt[:, i]) for i in range(bt.shape[1])])
    return [any(bt[:, i]) for i in range(bt.shape[1])]

def get_atom_featurizer(feature_mode, element_list):
    atom_mass_fun = chemutils.ConcatFeaturizer(
        [chemutils.atom_mass]
    )
    
    def atom_type_one_hot(atom):
        return chemutils.atom_type_one_hot(
            atom, allowable_set = element_list, encode_unknown = True
        )
    
    if feature_mode == 'light':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot
        ])
    elif feature_mode == 'full':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot, 
            atom_bond_type_one_hot,
            chemutils.atom_degree_one_hot, 
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_explicit_valence_one_hot,
            chemutils.atom_implicit_valence_one_hot,
            chemutils.atom_hybridization_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_formal_charge_one_hot,
            chemutils.atom_num_radical_electrons_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot,
            chemutils.atom_chiral_tag_one_hot
        ])
    elif feature_mode == 'medium':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot, 
            atom_bond_type_one_hot,
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot,
        ])

    return chemutils.BaseAtomFeaturizer(
        {"h": atom_featurizer_funs, 
        "m": atom_mass_fun}
    )

def get_bond_featurizer(feature_mode, self_loop):
    if feature_mode == 'light':
        return chemutils.BaseBondFeaturizer(
            featurizer_funcs = {'e': chemutils.ConcatFeaturizer([
                chemutils.bond_type_one_hot
            ])}, self_loop = self_loop
        )
    elif feature_mode == 'full':
        return chemutils.CanonicalBondFeaturizer(
            bond_data_field='e', self_loop = self_loop
        )
    
def create_atoms(mol):
    # NOTE: my error handling
    try:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    except Exception as e:
        print("Error creating atoms: {}".format(str(e)))
        return None
    return np.array(atoms)

def get_atom_scaffold(mol):
    if mol == None:
        return None
    # Get the Murcko scaffold of the molecule
    scaffold = GetScaffoldForMol(mol)
    if len(scaffold.GetAtoms()) == 0:
        return mol
    # scaffold = Chem.RWMol(scaffold)

    # # Count atoms in the original molecule and the scaffold
    # c1 = Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
    # c2 = Counter([atom.GetSymbol() for atom in scaffold.GetAtoms()])

    # # Subtract counts
    # result_counter = c1 - c2

    # # Add missing atoms to the scaffold
    # for atom_symbol in result_counter.elements():
    #     new_atom = Chem.Atom(atom_symbol)
    #     scaffold.AddAtom(new_atom)

    # # Convert RWMol back to Mol
    # scaffold = scaffold.GetMol()
    # print(Chem.MolToSmiles(scaffold))
    return scaffold

def mol_to_graph(mol, inchi, molgraph_dict, params, device):
    smile = Chem.MolToSmiles(mol)
    atoms = create_atoms(mol)
    
    if atoms is None:
        print("No atom failure in compound {}".format(smile))
        return False

    if len(atoms) == 1:
        print("Found unit length compound {}".format(smile))
        return False
    
    atom_in_list = [a in params['element_list'] for a in atoms]            
    if not all(atom_in_list):
        print(atoms[~np.array(atom_in_list)])
        print('Atom not in list')
        return False

    if inchi not in molgraph_dict.keys():
        node_featurizer = get_atom_featurizer(params['atom_feature'], params['element_list']) 
        edge_featurizer = get_bond_featurizer(params['bond_feature'], True)
        
        # Get the connected components of the molecule
        # fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        
        # # Create individual graphs for each fragment
        # graphs = []
        # for frag in fragments:
        #     Chem.SanitizeMol(frag) 
        #     g = chemutils.mol_to_bigraph(frag, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, add_self_loop=True,
        #                                  num_virtual_nodes=0)
        #     if g is not None:
        #         graphs.append(g)
        
        # if not graphs:
        #     print("Could not create graph for {}".format(smile))
        #     return False
        
        # # Combine the individual graphs into one
        # combined_graph = chemutils.combine_graphs(graphs)
        try:
            Chem.SanitizeMol(mol) 
        except:
            print('Invalid Molecule')
            return False
        g = chemutils.mol_to_bigraph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, add_self_loop=True,
                                     num_virtual_nodes=0)
        if g is None:
            print("Could not create graph for {}".format(smile))
            return False
        molgraph_dict[inchi] = g.to(device)
    return True
    
def single_molgraph_return(ik, mg_dict, cand_mg_dict, mol_dict, params, device):
    ret_mg = mg_dict.get(ik, None)
    if ret_mg != None:
        return ret_mg
    mol = mol_dict.get(ik, None)
    if mol == None:
        assert "No mol found!"
    else:
        if not mol_to_graph(mol, ik, cand_mg_dict, params, device):
            assert "could not create mol graph"
        ret_mg = cand_mg_dict[ik]
    
    return ret_mg

def single_molgraph(ik, mg_dict, mol_dict, params, device):
    mol = mol_dict.get(ik, None)
    if mol == None:
        return False
    else:
        if not mol_to_graph(mol, ik, mg_dict, params, device):
            print("No graph1 for this!! {}".format(ik))
            return False
    return True

def update_cand_list(target_ik, cand_list, mg_dict, mol_dict, cdict, params, device):
    new_cand_list = cand_list.copy()
    for cand in cand_list:
        cand_ik = cand[0]
        if not single_molgraph(cand_ik, mg_dict, mol_dict, params, device):
            new_cand_list.remove(cand)
    if len(new_cand_list) == 0:
        mol = mol_dict[target_ik]
        fp_q = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096,bitInfo={})
        while True:
            dum_ik = random.choice(list(cdict.keys()))
            if dum_ik == target_ik:
                continue
            dum_cand = cdict[dum_ik][0][0]
            if dum_cand == target_ik:
                continue
            if not single_molgraph(dum_cand, mg_dict, mol_dict, params, device):
                continue
            mol_cand = mol_dict[dum_cand]
            fp_cand_dum = GetMorganFingerprintAsBitVect(mol_cand, radius=2, nBits=4096,bitInfo={})
            sim = DataStructs.TanimotoSimilarity(fp_q, fp_cand_dum)
            new_cand_list = [(dum_cand, sim)]
            break
    return new_cand_list

def load_augmented_data(contr_data_dict_train, molgraph_dict, mol_dict, params, device):
    for k,v in tqdm(contr_data_dict_train.items()):
        new_v = v.copy()
        for item in v:
            inchi = item[0]
            mol = mol_dict.get(inchi, None)
            if mol == None:
                new_v.remove(item)
            else:
                if not mol_to_graph(mol, inchi, molgraph_dict, params, device):
                    print("No graph2 for this!! {}".format(inchi))
                    new_v.remove(item)
        if len(new_v) == 0:
            new_v = [(k, 1.0)] #if all cands gone, add a dummy cand
        contr_data_dict_train[k] = new_v

def sort_augmented_data(contr_data_dict_train, molgraph_dict, mol_dict, params, device):
    sorted_cand_dict_train = {}
    for ik, cand_l in contr_data_dict_train.items():
        cand_l.sort(key=lambda x: x[1], reverse=True)
        sorted_cand_dict_train[ik] = [0, cand_l]
    
    return sorted_cand_dict_train

def load_contrastive_data(dataset_builder, params, device, split=None):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    contr_data_dict = defaultdict(lambda: [[], []]) #inchi, [m/z, intensity]
    cnt = 0
    negcnt = 0
    nogrcnt = 0
    dataset_builder.cand_molgraph_dict = {}
    load_dicts = params['load_dicts']
    spec_type = params['spec_type']
    split_dict = dataset_builder.split_dict
    smi_dict = {}
    for k,v in data_dict.items():

        if v['Precursor'] not in spec_type:
            negcnt += 1
            continue
        
        if load_dicts:
            inchi = v['inchikey']
        else:
            inchi = v['smiles']
            
        if split is not None:
            if inchi not in split_dict[split]:
                continue
        elif params['ignore_test_contr']:
            if inchi in split_dict['test']:
                continue
            
        cnt += 1
        if params['debug'] and cnt > 100:
            break
        
        if load_dicts:
            inchi = v['inchikey']
            mol = get_atom_scaffold(mol_dict.get(inchi, None))
            smi = Chem.MolToSmiles(mol)
            if smi not in smi_dict:
                smi_dict[smi] = inchi
            else:
                inchi = smi_dict[smi]
        else:
            inchi = v['smiles']
            mol = get_atom_scaffold(Chem.MolFromSmiles(inchi))
            inchi = Chem.MolToSmiles(mol)
            
        molid = k
        if mol == None:
            continue
        
        if not mol_to_graph(mol, inchi, molgraph_dict, params, device):
            print("No graph3 for this!! {}".format(inchi))
            nogrcnt += 1
            continue
        
        #spectra_repr = get_sinus_repr(v, params)
        msi = v['ms']
        #fp = fp_dict[inchi].toarray()[0]
        fp = [0.0] * 4096 #Hack
        contr_data_dict[inchi][0].append((inchi, fp))
        #contr_data_dict[inchi][0].append(fp)
        contr_data_dict[inchi][1].append(msi)
        #contr_data_dict[inchi][1].append(msi[:,0])
        
    key_list = sorted(list(contr_data_dict.keys()))
    print("negcnt = {} nogrcnt = {}".format(negcnt, nogrcnt))
    
    if params['aug_cands']:
        cand_dict_train = dataset_builder.cand_dict_train
        sorted_cand_dict_train = sort_augmented_data(cand_dict_train, molgraph_dict, mol_dict, params, device)
        dataset_builder.cand_dict_train = sorted_cand_dict_train
        
    return contr_data_dict, key_list

def load_spectra_data(dataset_builder, params, dtype, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    split_dict = dataset_builder.split_dict
    in_to_id_dict = dataset_builder.in_to_id_dict
    inchis = split_dict[dtype]
    ret_list = []
    cnt = 0
    load_dicts = params['load_dicts']
    
    # if load_dicts:
    #     mol = mol_dict.get(ik, None)
    # else:
    #     mol = get_atom_scaffold(Chem.MolFromSmiles(ik))

    for inchi in inchis:
        if params['debug'] and cnt > 100:
            break
        if load_dicts:
            mol = get_atom_scaffold(mol_dict.get(inchi, None))
        else:
            mol = get_atom_scaffold(Chem.MolFromSmiles(inchi))
        
        if not mol:
            continue
        
        if not mol_to_graph(mol, inchi, molgraph_dict, params, device):
            print("No graph4 for this!! {}".format(inchi))
            continue

        ids = in_to_id_dict[inchi]
        for idx in ids:
            v = data_dict[idx]
            if v['Precursor'] != '[M+H]+':
                continue
            cnt += 1
            if params['debug'] and cnt > 100:
                break
            
            inchi2 = v['inchikey']
            mol = mol_dict.get(inchi2, None)
            if not mol:
                continue
            
            if not mol_to_graph(mol, inchi2, molgraph_dict, params, device):
                print("No graph5 for this!! {}".format(inchi2))
                continue
            
            msi = v['ms']
            #fp = fp_dict[inchi2].toarray()[0]
            fp = [0.0] * 4096 #Hack
            ret_list.append((inchi, msi, fp, 1))
    
    return ret_list

def load_spectra_data_single(dataset_builder, params, ik, id_list, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    in_to_id_dict = dataset_builder.in_to_id_dict
    ret_list = []
    cnt = 0
    load_dicts = params['load_dicts']
    spec_type = params['spec_type']
    
    if load_dicts:
        mol = get_atom_scaffold(mol_dict.get(ik, None))
    else:
        mol = get_atom_scaffold(Chem.MolFromSmiles(ik))
    
    if not mol:
        return
    
    if not mol_to_graph(mol, ik, molgraph_dict, params, device):
        print("No graph6 for this!! {}".format(ik))
        return

    for idx in id_list:
        v = data_dict[idx]
        if v['Precursor'] not in '[M+H]+':
            continue
        cnt += 1
        if params['debug'] and cnt > 100:
            break
        
        if load_dicts:
            inchi2 = v['inchikey']
            mol = get_atom_scaffold(mol_dict.get(inchi2, None))
        else:
            inchi2 = v['smiles']
            mol = get_atom_scaffold(Chem.MolFromSmiles(inchi2))
            
        if not mol:
            continue
        
        if not mol_to_graph(mol, inchi2, molgraph_dict, params, device):
            print("No graph7 for this!! {}".format(ik))
            continue
        
        msi = v['ms']
        #fp = fp_dict[inchi2].toarray()[0]
        fp = [0.0] * 4096 #Hack
        ret_list.append((ik, msi, fp, 1))

    return ret_list

def load_spectra_wneg_data(dataset_builder, params, dtype, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    split_dict = dataset_builder.split_dict
    in_to_id_dict_wneg = dataset_builder.in_to_id_dict_wneg
    inchis = split_dict[dtype]
    if dtype == 'train' and params['augment'] == True:
        data_dir = dataset_builder.data_dir
        f = open(data_dir + 'split_dict_aug.pkl', 'rb')
        split_dict_aug = pickle.load(f)
        inchis_aug = split_dict_aug['train']
        inchis += inchis_aug
    ret_list = []
    cnt = 0
    load_dicts = params['load_dicts']
    spec_type = params['spec_type']
    inchi_result = {}
    
    for inchi in tqdm(inchis):
        if inchi in inchi_result.keys():
            ret_list.append(inchi_result[inchi])
            continue
        if params['debug'] and cnt > 100:
            break
        if load_dicts:
            mol = get_atom_scaffold(mol_dict.get(inchi, None))
        else:
            mol = get_atom_scaffold(Chem.MolFromSmiles(inchi))
            
        if not mol:
            continue
        
        # if not mol_to_graph(mol=mol, inchi, molgraph_dict, params, device):
        #     print("No graph8 for this!! {}".format(inchi))
        #     continue
        skip_list = []
        ids = in_to_id_dict_wneg[inchi]
        for id_gt in ids:
            idx = id_gt[0]
            v = data_dict[idx]
            if v['Precursor'] not in spec_type:
                continue
            cnt += 1
            if params['debug'] and cnt > 100:
                break
            
            if load_dicts:
                inchi2 = v['inchikey']
                if inchi2 in skip_list:
                    continue
                mol = get_atom_scaffold(mol_dict.get(inchi2, None))
            else:
                inchi2 = v['smiles']
                mol = get_atom_scaffold(Chem.MolFromSmiles(inchi2))
                
            if not mol:
                continue
            
            if not mol_to_graph(mol, inchi2, molgraph_dict, params, device):
                skip_list.append(inchi2)
                print("No graph9 for this!! {}".format(inchi2))
                continue
            
            msi = v['ms']
            fp = [0.0] * 4096 #Hack
            gt = id_gt[1]
            ret_list.append((inchi, msi, fp, gt))
            inchi_result[inchi] = (inchi, msi, fp, gt)
    
    return ret_list

def load_cand_test_data(dataset_builder, params, ik, cand_list, spec, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    ret_list = []
    load_dicts = params['load_dicts']
    spec_type = params['spec_type']
    smi_dict = {}
    if load_dicts:
        mol = get_atom_scaffold(mol_dict.get(ik, None))
    else:
        mol = get_atom_scaffold(Chem.MolFromSmiles(ik))
        
    if not mol:
        return ret_list
    
    if not mol_to_graph(mol, ik, molgraph_dict, params, device):
        print("No graph10 for this!! {}".format(ik))
        return ret_list
    
    if ik in cand_list:
        cand_list.remove(ik)
    # iklist = [ik] + cand_list
    iklist = cand_list
    v = data_dict.get(spec, None)
    if v == None:
        return ret_list
    
    if v['Precursor'] not in spec_type:
        #print("Not a positive spectrum!!")
        return ret_list

    msi = v['ms']
    if load_dicts:
        inchi2 = v['inchikey']
        mol = get_atom_scaffold(mol_dict.get(inchi2, None))
        smi=Chem.MolToSmiles(mol)
        if smi not in smi_dict:
            smi_dict[smi] = inchi2
        else:
            inchi2 = smi_dict[smi]
    else:
        inchi2 = v['smiles']
        mol = get_atom_scaffold(Chem.MolFromSmiles(inchi2))
        inchi2 = Chem.MolToSmiles(mol)
        
    if not mol:
        print("No mol for this!! {}".format(inchi2))
        return ret_list
    
    if not mol_to_graph(mol, inchi2, molgraph_dict, params, device):
        print("No graph11 for this!! {}".format(inchi2))
        return ret_list
        
    #fp = fp_dict[inchi2].toarray()[0]
    fp = [0.0] * 4096 #Hack
    result_dict = {}
    for ikey in iklist:
        if ikey in result_dict.keys():
            ret_list.append(result_dict[ikey])
            continue
        if load_dicts:
            mol = get_atom_scaffold(mol_dict.get(ikey, None))
        else:
            mol = get_atom_scaffold(Chem.MolFromSmiles(ikey))
            ikey = Chem.MolToSmiles(mol)
            
        if not mol:
            continue
        
        if not mol_to_graph(mol, ikey, molgraph_dict, params, device):
            print("No graph12 for this!! {}".format(ikey))
            continue

        ret_list.append((ikey, msi, fp, 1))
        result_dict[ikey] = (ikey, msi, fp, 1)
    return ret_list

def load_vis_test_data(dataset_builder, params, ik_dict, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    ret_list = []
    
    for ik, spec in ik_dict.items():
        spec = spec[0]
        v = data_dict.get(spec, None)
        if v == None:
            continue
        
        if v['Precursor'] != '[M+H]+':
            #print("Not a positive spectrum!!")
            continue

        msi = v['ms']
        mol = mol_dict.get(ik, None)
        if not mol:
            print("No mol for this!! {}".format(ik))
            continue
        
        if not mol_to_graph(mol, ik, molgraph_dict, params, device):
            print("No graph13 for this!! {}".format(ik))
            continue
            
        #fp = fp_dict[inchi2].toarray()[0]
        fp = [0.0] * 4096 #Hack   
        
        ret_list.append((ik, msi, fp, 1))
                
    return ret_list

def load_vis_test_data2(dataset_builder, params, ik_dict, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    ret_list = []
    
    for ik, specs in ik_dict.items():
        for spec in specs:
            v = data_dict.get(spec, None)
            if v == None:
                print("not found spec!!")
                continue
            
            if v['Precursor'] != '[M+H]+':
                #print("Not a positive spectrum!!")
                continue
    
            msi = v['ms']
            mol = mol_dict.get(ik, None)
            if not mol:
                print("No mol for this!! {}".format(ik))
                continue
            
            if not mol_to_graph(mol, ik, molgraph_dict, params, device):
                print("No graph14 for this!! {}".format(ik))
                continue
                
            #fp = fp_dict[inchi2].toarray()[0]
            fp = [0.0] * 4096 #Hack   
            
            ret_list.append((ik, msi, fp, 1))
                
    return ret_list

if __name__ == "__main__":
    pass