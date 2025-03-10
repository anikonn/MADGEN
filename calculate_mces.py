import pandas as pd
import argparse
from rdkit import Chem
from rdkit import RDLogger
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from myopic_mces.myopic_mces import MCES

# Suppress all logs
os.environ['HIGHS_OUTPUT'] = 'OFF'
RDLogger.DisableLog('rdApp.*')
logging.getLogger('pulp').setLevel(logging.ERROR)
logging.getLogger('highspy').setLevel(logging.ERROR)

class MyopicMCES:
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.solver = 'HiGHS'
        self.solver_options = {
            'msg': 0,
            'log_to_console': False,
            'output_flag': False,
            'time_limit': 30,
            'log_file': os.devnull,
            'highs_debug_level': 0,
            'highs_verbosity': 'off'
        }

    def __call__(self, smiles_1, smiles_2):
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            threshold=self.threshold,
            solver=self.solver,
            solver_options=self.solver_options,
            ind=0
        )
        dist = retval[1]
        return dist

def calculate_mces_batch(args):
    mces, pairs = args
    results = {}
    for smiles_1, smiles_2 in pairs:
        results[(smiles_1, smiles_2)] = mces(smiles_1, smiles_2)
    return results

def calculate_mces_parallel(mces, pairs, n_workers=None):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    total_pairs = len(pairs)
    chunk_size = max(1, total_pairs // n_workers)
    pair_chunks = [pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
    
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_chunk = {
            executor.submit(calculate_mces_batch, (mces, chunk)): chunk 
            for chunk in pair_chunks
        }
        for future in as_completed(future_to_chunk):
            results.update(future.result())
    return results

def process_csv(file_path, k):
    df = pd.read_csv(file_path)
    myopic_mces = MyopicMCES()
    mces_cache = {}
    results = []

    if k == 1:
        # Process in chunks of 10, only first row's true and pred
        for i in tqdm(range(0, len(df), 10)):
            chunk = df.iloc[i:i+10]
            true_smile = chunk.iloc[0]['true']
            for j in range(chunk.shape[0]):
                pred_smile = chunk.iloc[j]['pred']
                if Chem.MolFromSmiles(pred_smile) is not None:
                    break
            if (true_smile, pred_smile) not in mces_cache:
                pairs = [(true_smile, pred_smile)]
                new_results = calculate_mces_parallel(myopic_mces, pairs)
                mces_cache.update(new_results)
            
            mces_value = mces_cache[(true_smile, pred_smile)]
            results.append({
                'chunk_start_idx': i,
                'true': true_smile,
                'pred': pred_smile,
                'mces': mces_value
            })
    
    else:  # k == 10
        # Process every row's true and pred
        pairs = []
        for idx, row in df.iterrows():
            true_smile = row['true']
            pred_smile = row['pred']
            if (true_smile, pred_smile) not in mces_cache:
                pairs.append((true_smile, pred_smile))
        
        if pairs:
            new_results = calculate_mces_parallel(myopic_mces, pairs)
            mces_cache.update(new_results)
        
        for idx, row in df.iterrows():
            true_smile = row['true']
            pred_smile = row['pred']
            mces_value = mces_cache[(true_smile, pred_smile)]
            results.append({
                'idx': idx,
                'true': true_smile,
                'pred': pred_smile,
                'mces': mces_value
            })

    # Save results
    output_df = pd.DataFrame(results)
    output_path = f"{os.path.splitext(file_path)[0]}_mces_k{k}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MCES for CSV files")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--k', type=int, choices=[1, 10], required=True, help='Processing mode (1 or 10)')
    args = parser.parse_args()

    process_csv(args.file_path, args.k)