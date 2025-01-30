import pickle
import pandas as pd

split_dict = {'train':[], 'valid':[], 'test': []}

smile_id = pickle.load(open('/data2/apurva/spectra/canopus/inchi_to_id_dict.pkl','rb'))
data = pd.read_csv('/data2/yinkai/RetroBridge/data/canopus/raw/canopus.csv')
id_inchi = dict()
for item in smile_id.items():
    for id1 in item[1]:
        if id1 in id_inchi.keys():
            print(item)
        id_inchi[id1] = item[0]
# id_inchi = {item[1]:item[0] for item in smile_id.items()}
for index, row in data.iterrows():
    split_dict[row['source'].replace('val','valid')].append(id_inchi[row['key']])
    
pickle.dump(split_dict, open('./split.pkl','wb'))
