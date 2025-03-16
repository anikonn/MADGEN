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
# 1740431736104 4.13% (716/17320)
# 1740431803101 12.42%
# 1740508838828 canopus 5.31% (39/735)
sca_list = pickle.load(open('./ranks_total_1741662754375.pkl', 'rb'))
mol_dict = pickle.load(open('/data/yinkai/canopus/mol_dict.pkl', 'rb'))
correct_count = 0
total_count = 0
sca_dict = {}
k = 20
for ele in sca_list:
    # query = mol_dict.get(ele[0])  # Query molecule
    query = ele[0]  # Query molecule
    # predictions = [mol_dict.get(i) for i in ele[4][:k]]  # First prediction
    predictions = ele[4][:k]  # First prediction
    for prediction in predictions:
        if query not in sca_dict.keys():
            sca_dict[query] = {}
        if prediction not in sca_dict[query].keys():
            sca_dict[query][prediction] = 1
        else:
            sca_dict[query][prediction] += 1

# for i in range(len(sca_list)):
#     if get_sca_smile(sca_list[i][0]) in [get_sca_smile(p) for p in sca_list[i][4]]:
#         print(i)
#         break
# for i in range(len(sca_list[9])):
#     if get_sca_smile(sca_list[9][0]) == get_sca_smile(sca_list[9][4][i]):
#         print(i)

# count = 0
# for i in sca_dict.keys():
#     if get_sca_smile(i) in [get_sca_smile(k) for k in list(sca_dict[i].keys())[:1]]:
#         count += 1
# # print(count)

for i in sca_dict.keys():
    sca_dict[i] = {k: v for k, v in sorted(sca_dict[i].items(), key=lambda item: item[1], reverse=True)}

# count = 0
# for i in sca_list:
#     if get_sca_smile(mol_dict.get(i[0])) == get_sca_smile(list(sca_dict[mol_dict.get(i[0])].keys())[0]):
#         count +=1
result_dict = {}
count = 0
for i in sca_list:
    result_dict[i[1]] = get_sca_smile(list(sca_dict[i[0]].keys())[0])

pickle.dump(result_dict, open('./ranks_msgym_pred.pkl', 'wb'))
