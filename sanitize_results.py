#!/usr/bin/env python3
# sanitize_results.py
# Usage:
#   python sanitize_results.py --file_path table.csv

import argparse
import re
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import RemoveHs

RDLogger.DisableLog("rdApp.*")
_lf = rdMolStandardize.LargestFragmentChooser()
_WS_RE = re.compile(r"[\u00A0\u2007\u202F]")  # common non-breaking spaces

def clean_smiles(raw: str, fix_nitro: bool = True):
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'")
    # normalize spaces
    s = _WS_RE.sub(" ", s).replace(" ", "")
    if not s:
        return None
    if fix_nitro:
        # replace bare N(=O)O with [N+](=O)[O-] when not already bracketed
        s = re.sub(r'(?<!\[)N\(=O\)O(?!\])', r'[N+](=O)[O-]', s)
    return s

def make_valid_mol(smi: str):
    """
    Try to produce a sanitized RDKit Mol.
    Returns (mol or None, reason_if_failed or "").
    """
    s = clean_smiles(smi, fix_nitro=True)
    if not s:
        return None, "Empty/None after cleaning"

    # first pass: parse without sanitize, then sanitize explicitly
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return None, "MolFromSmilesFailed"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"SanitizeError: {e}"

    # optional normalizations: largest fragment, remove Hs
    try:
        mol = _lf.choose(mol)
    except Exception:
        pass
    try:
        mol = RemoveHs(mol)
    except Exception:
        pass

    return mol, ""

def main():
    ap = argparse.ArgumentParser(description="Sanitize and (lightly) fix invalid SMILES in result table.")
    ap.add_argument("--file_path", required=True, help="CSV with columns incl. 'pred'")
    ap.add_argument("--keep_reason", action="store_true", help="Keep reason column for debugging")
    args = ap.parse_args()

    df = pd.read_csv(args.file_path)
    if "pred" not in df.columns:
        raise SystemExit("Input CSV must have a 'pred' column.")

    val_flags, reasons = [], []
    for smi in df["pred"].tolist():
        mol, why = make_valid_mol(smi)
        val_flags.append(mol is not None)
        reasons.append("" if mol is not None else why)

    df["pred_valid"] = val_flags
    df["invalid_reason"] = reasons

    before = len(df)
    df_clean = df[df["pred_valid"]].copy()
    after = len(df_clean)
    print(f"[info] Kept {after}/{before} rows ({before-after} invalid removed).")

    if not args.keep_reason:
        df_clean.drop(columns=["invalid_reason"], inplace=True, errors="ignore")

    out_path = args.file_path.replace(".csv", "_clean.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"[ok] Saved cleaned table -> {out_path}")

if __name__ == "__main__":
    main()
