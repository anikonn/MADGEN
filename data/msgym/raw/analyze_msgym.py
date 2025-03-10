import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import csv

# Load data
ms_dict = pickle.load(open('./data/msgym/raw/msgym.pkl', 'rb'))
data = pd.DataFrame(ms_dict)
smiles_list = list(data['smiles'])

# Initialize lists to store all data
molecule_data = []
scaffold_data = {}  # Using dict to avoid duplicate scaffolds

print("Processing molecules...")
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    # Get scaffold first
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
    
    # Calculate free atoms (atoms in molecule but not in scaffold)
    n_free_atoms = mol.GetNumAtoms() - scaffold.GetNumAtoms() if scaffold else mol.GetNumAtoms()
    
    # Get molecule stats
    mol_stats = {
        'smiles': smiles,
        'n_nodes': mol.GetNumAtoms(),
        'n_edges': mol.GetNumBonds(),
        'n_free_atoms': n_free_atoms
    }
    molecule_data.append(mol_stats)
    
    # Get scaffold stats
    if scaffold:
        if scaffold_smiles not in scaffold_data:
            scaffold_data[scaffold_smiles] = {
                'smiles': scaffold_smiles,
                'n_nodes': scaffold.GetNumAtoms(),
                'n_edges': scaffold.GetNumBonds(),
                'frequency': 1
            }
        else:
            scaffold_data[scaffold_smiles]['frequency'] += 1

# Save molecule statistics
with open('molecule_statistics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['smiles', 'n_nodes', 'n_edges', 'n_free_atoms'])
    writer.writeheader()
    writer.writerows(molecule_data)

# Save scaffold statistics
with open('scaffold_statistics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['smiles', 'n_nodes', 'n_edges', 'frequency'])
    writer.writeheader()
    writer.writerows(scaffold_data.values())

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of molecules: {len(molecule_data)}")
print(f"Total number of unique scaffolds: {len(scaffold_data)}")

# Calculate averages
avg_mol_nodes = sum(m['n_nodes'] for m in molecule_data) / len(molecule_data)
avg_mol_edges = sum(m['n_edges'] for m in molecule_data) / len(molecule_data)
avg_mol_free = sum(m['n_free_atoms'] for m in molecule_data) / len(molecule_data)
avg_scaf_nodes = sum(s['n_nodes'] for s in scaffold_data.values()) / len(scaffold_data)
avg_scaf_edges = sum(s['n_edges'] for s in scaffold_data.values()) / len(scaffold_data)

print(f"\nAverages:")
print(f"Average nodes per molecule: {avg_mol_nodes:.2f}")
print(f"Average edges per molecule: {avg_mol_edges:.2f}")
print(f"Average free atoms per molecule: {avg_mol_free:.2f}")
print(f"Average nodes per scaffold: {avg_scaf_nodes:.2f}")
print(f"Average edges per scaffold: {avg_scaf_edges:.2f}")