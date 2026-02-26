#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smi: str):
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def has_scaffold(pred_mol, scaffold_mol) -> bool:
    if pred_mol is None or scaffold_mol is None:
        return False
    try:
        return pred_mol.HasSubstructMatch(scaffold_mol)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Compute scaffold coverage on generated samples and save per-molecule plots.")
    ap.add_argument("--input_csv", required=True, help="CSV from sample.py (columns: scaffold,pred,true,score,nll,ell)")
    ap.add_argument("--out_csv", default="scaffold_coverage_summary.csv", help="Where to write per-molecule summary CSV")
    ap.add_argument("--plots_dir", default="scaffold_plots", help="Directory to write true_vs_scaffold images")
    ap.add_argument("--max_per_mol", type=int, default=None, help="Optionally cap number of preds per molecule (e.g., 100)")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    # Basic sanity columns
    needed = {"scaffold", "pred", "true"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Group by (true, scaffold) — in msgym runs, each true has a single scaffold
    groups = df.groupby(["true", "scaffold"], sort=False)

    rows = []
    overall_n_molecules = 0
    overall_valid_preds = 0
    overall_with_scaffold = 0

    print(f"Found {len(groups)} molecules in the table")
    for (true_smi, scaffold_smi), g in tqdm(groups, desc="Evaluating", unit="mol"):
        print(1)
        # Optionally cap number of preds
        preds = g["pred"].tolist()
        if args.max_per_mol is not None:
            preds = preds[: args.max_per_mol]

        true_mol = mol_from_smiles(true_smi)
        scaf_mol = mol_from_smiles(scaffold_smi)

        # Some scaffolds in data can be multi-fragment ("A.B"). You can decide to skip or try the largest fragment.
        # Here: if multi-fragment, we try the **largest** fragment.
        if scaf_mol is None and isinstance(scaffold_smi, str) and "." in scaffold_smi:
            try:
                parts = scaffold_smi.split(".")
                parts_mols = [mol_from_smiles(p) for p in parts]
                parts_mols = [m for m in parts_mols if m is not None]
                if parts_mols:
                    scaf_mol = max(parts_mols, key=lambda m: m.GetNumAtoms())
            except Exception:
                pass

        # Count valid preds + how many contain scaffold
        valid_pred_mols = []
        n_with_scaf = 0
        for ps in preds:
            pm = mol_from_smiles(ps)
            if pm is None:
                continue
            valid_pred_mols.append(pm)
            if has_scaffold(pm, scaf_mol):
                n_with_scaf += 1

        n_preds = len(preds)
        n_valid = len(valid_pred_mols)
        frac_scaf = (n_with_scaf / n_valid) if n_valid > 0 else 0.0

        rows.append(
            {
                "true": true_smi,
                "scaffold": scaffold_smi,
                "n_rows_in_csv": len(g),            # raw rows for this molecule in input
                "n_preds_considered": n_preds,      # after optional cap
                "n_valid_preds": n_valid,           # RDKit-parsable preds
                "n_with_scaffold": n_with_scaf,     # substructure match
                "frac_with_scaffold": frac_scaf,
                "true_num_atoms": (true_mol.GetNumAtoms() if true_mol else None),
                "scaffold_num_atoms": (scaf_mol.GetNumAtoms() if scaf_mol else None),
            }
        )

        # Save side-by-side plot of true vs scaffold
        # If either is None, we still try to draw what we can.
        try:
            mols = []
            legends = []
            if true_mol is not None:
                mols.append(true_mol)
                legends.append("True")
            if scaf_mol is not None:
                mols.append(scaf_mol)
                legends.append("Scaffold")
            if mols:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=len(mols),
                    subImgSize=(350, 350),
                    legends=legends,
                )
                # filename from first 32 chars of SMILES to keep paths manageable
                base = f"{hash(true_smi) & 0xffffffff:x}"
                img_path = os.path.join(args.plots_dir, f"true_vs_scaffold_{base}.png")
                img.save(img_path)
        except Exception:
            pass

        # aggregate
        overall_n_molecules += 1
        overall_valid_preds += n_valid
        overall_with_scaffold += n_with_scaf

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_csv, index=False)

    print("\n==== Summary ====")
    print(f"Molecules evaluated: {overall_n_molecules}")
    print(f"Valid predictions total: {overall_valid_preds}")
    print(f"Predictions containing scaffold: {overall_with_scaffold}")
    overall_frac = (overall_with_scaffold / overall_valid_preds) if overall_valid_preds > 0 else 0.0
    print(f"Overall fraction (contains scaffold | valid preds): {overall_frac:.4f}")
    print(f"Per-molecule summary written to: {args.out_csv}")
    print(f"Per-molecule plots saved to: {args.plots_dir}")

    # === Histogram ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.hist(summary["frac_with_scaffold"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Fraction of predictions containing scaffold")
    plt.ylabel("Number of molecules")
    plt.title("Distribution of scaffold coverage per molecule")
    hist_path = os.path.join(args.plots_dir, "scaffold_fraction_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")


if __name__ == "__main__":
    main()
