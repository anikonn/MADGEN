import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm
from collections import defaultdict

def get_sca_smile(smiles):
    if type(smiles) == str:
        molecule = Chem.MolFromSmiles(smiles)
    else:
        molecule = smiles
    scaffold = GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)

# Load data
print("Loading data...")
sca_list = pickle.load(open('./ranks_total_1741662754375.pkl', 'rb'))
mol_dict = pickle.load(open('./mol_dict.pkl', 'rb'))

# Generate final results for k=110 (or whatever k you want to use)
k = 110
sca_dict = {}

print(f"Generating final results for k={k}...")
for ele in sca_list:
    query = mol_dict.get(ele[0])
    
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    predictions = [get_sca_smile(mol_dict.get(i)) for i in ele[4][:k] if mol_dict.get(i) is not None]
    
    for prediction in predictions:
        if formula not in sca_dict.keys():
            sca_dict[formula] = {}
        if prediction not in sca_dict[formula].keys():
            sca_dict[formula][prediction] = 1
        else:
            sca_dict[formula][prediction] += 1

# Sort predictions by frequency for each formula
for i in sca_dict.keys():
    sca_dict[i] = {k: v for k, v in sorted(sca_dict[i].items(), key=lambda item: item[1], reverse=True)}

# Generate result dictionary
result_dict = {}
for i in sca_list:
    query = mol_dict.get(i[0])
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    if formula in sca_dict and len(sca_dict[formula]) > 0:
        result_dict[i[1]] = list(sca_dict[formula].keys())[0]

# Calculate SPA (Scaffold Prediction Accuracy)
print("Calculating SPA...")
correct_count = 0
total_count = 0

for i in sca_list:
    query = mol_dict.get(i[0])
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    if formula in sca_dict and len(sca_dict[formula]) > 0:
        # Get predicted scaffold (most frequent scaffold for this formula)
        predicted_scaffold = list(sca_dict[formula].keys())[0]
        
        # Get true scaffold
        true_scaffold = get_sca_smile(query)
        
        # Compare predicted vs true scaffold
        if predicted_scaffold == true_scaffold:
            correct_count += 1
        total_count += 1

# Calculate and display SPA
spa = (correct_count / total_count) * 100 if total_count > 0 else 0
print(f"Scaffold Prediction Accuracy (SPA): {spa:.2f}% ({correct_count}/{total_count})")

print("Saving results...")
pickle.dump(result_dict, open('./ranks_canopus_pred.pkl', 'wb'))
print(f"Saved {len(result_dict)} results")
