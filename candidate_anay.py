import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm

def get_sca_smile(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    scaffold = GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)

sca_list = pickle.load(open('/cluster/tufts/liulab/yiwan01/MADGEN/JESTR1/results/ranks_total_1737484822765.pkl', 'rb'))
mol_dict = pickle.load(open('./data/canopus/raw/smiles_dict.pkl', 'rb'))
sca_dict = {}
correct_count = 0
total_count = 0

for ele in sca_list:
    query = mol_dict.get(ele[0])  # Query molecule
    prediction = mol_dict.get(ele[4][0])  # First prediction
    sca_dict[query] = prediction
    
    if get_sca_smile(query) == get_sca_smile(prediction):
        correct_count += 1
    total_count += 1

accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

