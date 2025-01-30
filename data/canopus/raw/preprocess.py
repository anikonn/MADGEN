import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm import tqdm

get_norm = True

# Load raw data
inchi_smiles = pickle.load(open('/cluster/tufts/liulab/yiwan01/MADGEN/data/canopus/raw/smile_to_id_dict.pkl', 'rb'))
with open('/cluster/tufts/liulab/yiwan01/MADGEN/data/canopus/raw/data_dict.pkl', 'rb') as f:
    inchi_ms = pickle.load(f)

# Process SMILES and MS data
id_smiles = {i[1][0]: i[0] for i in inchi_smiles.items()}
smiles_list = []
ms_list = []
identifier = []
length = []

for key in tqdm(inchi_ms.keys()):
    try:
        # Get SMILES and MS data
        smile = id_smiles[key]
        ms = inchi_ms[key]['ms']
        
        # Convert MS data to numpy array and reshape if needed
        ms_array = np.array(ms)
        
        # Store processed data
        smiles_list.append(smile)
        ms_list.append(ms_array)
        identifier.append(key)
        length.append(ms_array.shape[0])
    except:
        continue


# Normalize MS data if needed
if get_norm:
    spectra = [v for v in ms_list if len(np.array(v.tolist())) != 0]
    spectra = np.concatenate(spectra, axis=0)
    inten_mean = spectra[:, 0].mean()
    inten_std = spectra[:, 0].std()
    mz_mean = spectra[:, 1].mean()
    mz_std = spectra[:,1].std()
    print(f"Intensity - Mean: {spectra[:, 0].mean()}, Std: {spectra[:, 0].std()}")
    print(f"M/Z - Mean: {spectra[:, 1].mean()}, Std: {spectra[:, 1].std()}")
    
    # Normalize m/z values
    for i in range(len(ms_list)):
        ms_list[i][:, 0] = (ms_list[i][:, 0] - inten_mean) / inten_std
        ms_list[i][:, 1] = (ms_list[i][:, 1] - mz_mean) / mz_std

# Get scaffolds and create initial data
scaffolds = []
for smile in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        scaffolds.append(None)
        continue
    scaffold = MurckoScaffoldSmiles(mol=mol)
    scaffolds.append(scaffold if len(scaffold) > 3 else None)

# Create initial dataframe
data = {
    'smiles': smiles_list,
    'ms': ms_list,
    'scaffold': scaffolds,
    'identifier': identifier,
}

# Convert to DataFrame and shuffle
df = pd.DataFrame(data)
df = df.dropna(subset=['scaffold']).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Perform scaffold-based split
scaffold_counts = df.groupby('scaffold').size().reset_index(name='counts')
scaffold_counts_sorted = scaffold_counts.sort_values(by='counts', ascending=False)

train_scaffolds, val_scaffolds, test_scaffolds = [], [], []
train_count, val_count, test_count = 0, 0, 0

# Split scaffolds maintaining 8:1:1 ratio
for _, row in scaffold_counts_sorted.iterrows():
    scaffold, count = row['scaffold'], row['counts']
    if train_count <= 4 * (val_count + test_count):
        train_scaffolds.append(scaffold)
        train_count += count
    elif val_count <= test_count:
        val_scaffolds.append(scaffold)
        val_count += count
    else:
        test_scaffolds.append(scaffold)
        test_count += count

# Assign splits based on scaffolds
df['source'] = 'train'
df.loc[df['scaffold'].isin(val_scaffolds), 'source'] = 'val'
df.loc[df['scaffold'].isin(test_scaffolds), 'source'] = 'test'

print(f"Split sizes - Train: {train_count}, Val: {val_count}, Test: {test_count}")

# Process molecules and create final data
processed_data = {
    'smiles': [],
    'ms': [],
    'source': [],
    'identifier': []
}

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    molecule = Chem.MolFromSmiles(row['smiles'])
    if molecule is None:
        continue
    smile = Chem.MolToSmiles(molecule)
    
    processed_data['smiles'].append(smile)
    processed_data['ms'].append(row['ms'])
    processed_data['source'].append(row['source'])
    processed_data['identifier'].append(row['identifier'])

# Save processed data
with open('canopus.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print(f"Processed {len(processed_data['smiles'])} valid molecules")