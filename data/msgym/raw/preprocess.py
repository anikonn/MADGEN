import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Atom
from tqdm import tqdm
import math
get_norm = True

data = pd.read_csv('./MassSpecGym (2).tsv', sep='\t')

stages = data['fold']
smiles_list = data['smiles']   
inten =  [np.array(eval(i)) for i in data['intensities']] 
mz = [np.array(eval(i)) for i in data['mzs']]
ms_list = [np.array(i) for i in list(zip(inten, mz))]
energy = data['collision_energy']
instrument_type = data['instrument_type']   
adduct = data['adduct']
identifier = data['identifier']
length = []
for i in range(len(ms_list)):
    if ms_list[i].shape == (2,):
        ms_list[i] = ms_list[i].reshape(2, 1)
    try: 
        ms_list[i] = np.transpose(ms_list[i], (1,0))
    except:
        print(ms_list[i].shape, i, data.iloc[i])
    length.append(ms_list[i].shape[0])
plt.hist(length, bins='auto')
plt.title('Distribution of num of peaks')
plt.savefig('distribution_numPeaks.png')
if get_norm:
    spectra = [v for v in ms_list if len(np.array(v.tolist())) != 0 ]
    # print(len(spectra), spectra[4359])
    spectra = np.concatenate(spectra, axis=0)
    print(spectra[:, 0].mean()) # 0.10126370929621457
    print(spectra[:, 0].std()) # 0.22392237788824612
    print(spectra[:, 1].mean()) # 227.68274551635218
    print(spectra[:, 1].std()) # 170.81643093234982
    mean_mz = spectra[:, 1].mean()
    std_mz = spectra[:, 1].std()
    mean_inten = spectra[:, 0].mean()
    std_inten = spectra[:, 0].std()
    for i in range(len(ms_list)):
        # ms_list[i][:, 0] = (ms_list[i][:, 0]-0.10126370929621457)/0.22392237788824612
        ms_list[i][:, 1] = (ms_list[i][:, 1]-mean_mz)/std_mz
        # ms_list[i][:, 0] = (ms_list[i][:, 0]-mean_inten)/std_inten
data = {
'smiles': smiles_list,  
'ms': ms_list,
'source': stages,
'energy': energy,
'instrument_type': instrument_type,
'adduct': adduct,
'identifier': identifier,
}
print(ms_list[0])
ms = []
stage = []
smiles = []
energy = []
instrument_type = []
adduct = []
identifier = []
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
atom_symbols = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    molecule = Chem.MolFromSmiles(row['smiles'])
    smile = Chem.MolToSmiles(molecule)
    ms.append(row['ms'])
    stage.append(row['source'])
    smiles.append(smile)
    instrument_type.append(row['instrument_type'])
    adduct.append(row['adduct'])
    identifier.append(row['identifier'])

data = {
    'smiles': smiles,
    'ms': ms,
    'source': stage,
    'instrument_type': instrument_type,
    'adduct': adduct,
    'identifier': identifier,
    }
with open('msgym.pkl', 'wb') as f:
    pickle.dump(data, f)