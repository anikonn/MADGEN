import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm

def get_sca_smile(smiles):
    if type(smiles) == str:
        molecule = Chem.MolFromSmiles(smiles)
    else:
        molecule = smiles
    scaffold = GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)

# Load data
print("Loading data...")
sca_list = pickle.load(open('./ranks_total_1740431736104.pkl', 'rb'))

# Generate final results for k=30 (or whatever k you want to use)
k = 30
sca_dict = {}

print(f"Generating final results for k={k}...")
for ele in sca_list:
    query = Chem.MolFromSmiles(ele[0])  # Query molecule
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    # Convert predictions to scaffolds FIRST, then aggregate (like reference)
    predictions = [get_sca_smile(pred) for pred in ele[4][:k]]
    
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

# Calculate SPA (Scaffold Prediction Accuracy)
print("Calculating SPA...")
correct_count = 0
total_count = 0

for i in sca_list:
    query = Chem.MolFromSmiles(i[0])
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    if formula in sca_dict and len(sca_dict[formula]) > 0:
        # Get predicted scaffold (most frequent scaffold for this formula)
        predicted_scaffold = list(sca_dict[formula].keys())[0]
        
        # Get true scaffold
        true_scaffold = get_sca_smile(i[0])
        
        # Compare predicted vs true scaffold
        if predicted_scaffold == true_scaffold:
            correct_count += 1
        total_count += 1

# Calculate and display SPA
spa = (correct_count / total_count) * 100 if total_count > 0 else 0
print(f"Scaffold Prediction Accuracy (SPA): {spa:.2f}% ({correct_count}/{total_count})")

# Generate result dictionary
result_dict = {}
for i in sca_list:
    query = Chem.MolFromSmiles(i[0])
    if query is None:
        continue
        
    formula = Chem.rdMolDescriptors.CalcMolFormula(query)
    if formula in sca_dict and len(sca_dict[formula]) > 0:
        result_dict[i[1]] = list(sca_dict[formula].keys())[0]

print("Saving results...")
pickle.dump(result_dict, open('./ranks_msgym_pred.pkl', 'wb'))
print(f"Saved {len(result_dict)} results")

