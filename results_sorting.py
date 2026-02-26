import pandas as pd
from rdkit import Chem
import argparse

def num_atoms(smi):
    m = Chem.MolFromSmiles(smi)
    return m.GetNumAtoms() if m is not None else None

ap = argparse.ArgumentParser()
ap.add_argument("--file_path", required=True, help="CSV with columns: scaffold,pred,true,score,...")
args = ap.parse_args()

df = pd.read_csv(args.file_path)

df["total_ll"] = df["nll"] + df["ell"]

# preserve the original dataset order by assigning group IDs
df["group_id"] = (df["true"] + "|" + df["scaffold"]).factorize()[0]

# sort inside each group but keep global group order
df = df.sort_values(["group_id", "total_ll", "score"], ascending=[True, False, False])

# drop the helper column before saving
df = df.drop(columns="group_id")

df.to_csv(args.file_path.replace(".csv", "_my_ranked.csv"), index=False)