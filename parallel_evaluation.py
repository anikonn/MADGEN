import pulp
import pandas as pd
from tqdm import tqdm
import argparse

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

RDLogger.DisableLog("rdApp.*")

from myopic_mces.myopic_mces import MCES

class MyopicMCES:
    def __init__(
        self,
        ind=0,
        solver=pulp.listSolvers(onlyAvailable=True)[0],
        threshold=15,
        always_stronger_bound=True,
        solver_options=None,
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1, smiles_2):
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options,
        )
        dist = retval[1]
        return dist

def split_dataframe(df, chunk_size=10):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def calculate_mces(mces, pairs):
    """
    Run MCES computations for individual pairs in parallel.
    """
    results = {}
    for smiles_1, smiles_2 in pairs:
        if (smiles_1, smiles_2) not in results.keys():
            results[(smiles_1, smiles_2)] = mces(smiles_1, smiles_2)
    return results

def process_chunk(args):
    """
    Process a chunk of the dataframe with given parameters.
    """
    df, k, myopic_mces, mces_thld = args
    chunk_results = {"accuracy": 0, "similarity": 0, "MCES": 0}
    
    smile = list(df["true"])[0]

    pred_smiles = sorted(list(df["pred"]), key=lambda x: list(df["pred"]).count(x), reverse=True)
    pred_smiles = pred_smiles[:k]
    mol = Chem.MolFromSmiles(smile)
    
    if mol is None:
        return chunk_results
        
    pred_mols = [Chem.MolFromSmiles(pred) for pred in pred_smiles]
    in_top_k = MolToInchiKey(mol).split("-")[0] in [
        MolToInchiKey(pred).split("-")[0] if pred is not None else None
        for pred in pred_mols
    ]
    chunk_results["accuracy"] += int(in_top_k)
    
    # Calculate MCES
    dists = []
    pairs = [(smile, pred) for pred, pred_mol in zip(pred_smiles, pred_mols) if pred_mol is not None]
    mces_results = calculate_mces(myopic_mces, pairs)
    
    for pred, pred_mol in zip(pred_smiles, pred_mols):
        if pred_mol is None:
            dists.append(mces_thld)
        else:
            dists.append(mces_results.get((smile, pred), mces_thld))
    
    chunk_results["MCES"] += min(min(dists), mces_thld)
    
    # Calculate similarity
    mol_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    pred_fps = [
        GetMorganFingerprintAsBitVect(pred, radius=2, nBits=2048) if pred is not None else None 
        for pred in pred_mols
    ]
    sims = [
        TanimotoSimilarity(mol_fp, pred) if pred is not None else 0 for pred in pred_fps
    ]
    chunk_results["similarity"] += max(sims)
    
    return chunk_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes')
    args = parser.parse_args()

    # Replace the path with result path
    datasets = ['attention']
    for dataset in datasets:
        path = args.file_path
        df1 = pd.read_csv(path)
        result_metric = {"accuracy": 0, "similarity": 0, "MCES": 0}
        # Top K
        ks = [1, 10]
        mces_thld = 100
        myopic_mces = MyopicMCES(
            threshold=20,
            solver='HiGHS',
            solver_options={
                'msg': 0,
                'log_to_console': False,
                'output_flag': False,
                'time_limit': 10,
                'log_file': os.devnull,
                'highs_debug_level': 0,
                'highs_verbosity': 'off'
            }
        )
        
        for k in ks:
            sub_dfs = split_dataframe(df1, chunk_size=50)
            
            # Prepare arguments for parallel processing
            process_args = [(df, k, myopic_mces, mces_thld) for df in sub_dfs]
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                results = list(tqdm(
                    executor.map(process_chunk, process_args),
                    total=len(sub_dfs),
                    desc=f"Processing k={k}"
                ))
            
            # Aggregate results
            for chunk_result in results:
                for key in result_metric:
                    result_metric[key] += chunk_result[key]
            
            # Calculate averages
            for key in result_metric:
                result_metric[key] = result_metric[key] / len(sub_dfs)
            
            print(dataset, k, result_metric)
        print(result_metric)