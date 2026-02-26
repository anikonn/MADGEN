#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ------------------------- MCES wrapper (lazy import) -------------------------
class MyopicMCES:
    def __init__(
        self,
        ind=0,
        solver=None,               # let myopic_mces pick default if None
        threshold=20,
        always_stronger_bound=True,
        solver_options=None,
    ):
        # import here so script runs even if MCES is disabled / not installed
        from myopic_mces.myopic_mces import MCES as _MCES
        import pulp

        self._MCES = _MCES
        self.ind = ind
        # choose an available solver if none given
        self.solver = solver if solver is not None else pulp.listSolvers(onlyAvailable=True)[0]
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)
        self.solver_options = solver_options

    def __call__(self, s1, s2):
        try:
            _, dist = self._MCES(
                s1=s1,
                s2=s2,
                ind=self.ind,
                threshold=self.threshold,
                always_stronger_bound=self.always_stronger_bound,
                solver=self.solver,
                solver_options=self.solver_options,
            )
            return float(dist)
        except Exception:
            return float(self.threshold)


# ------------------------- chemistry helpers ---------------------------------
_lf = rdMolStandardize.LargestFragmentChooser()

def mol_from_smiles(smi: str):
    if not isinstance(smi, str):
        return None
    try:
        m = Chem.MolFromSmiles(smi)
    except Exception:
        return None
    if m is None:
        return None
    try:
        m = _lf.choose(m)  # keep largest fragment
    except Exception:
        pass
    try:
        m = RemoveHs(m)
    except Exception:
        pass
    return m

def inchi_block(mol):
    try:
        return Chem.inchi.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None

def has_scaffold(pred_mol, scaffold_mol, use_chirality=False):
    if pred_mol is None or scaffold_mol is None:
        return False
    try:
        return pred_mol.HasSubstructMatch(scaffold_mol, useChirality=use_chirality)
    except Exception:
        return False


# ------------------------- metrics -------------------------------------------
def tanimoto_max(mol, pred_mols):
    if mol is None or not pred_mols:
        return 0.0
    fp = rdmd.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sims = []
    for pm in pred_mols:
        if pm is None:
            sims.append(0.0)
        else:
            pf = rdmd.GetMorganFingerprintAsBitVect(pm, radius=2, nBits=2048)
            sims.append(DataStructs.TanimotoSimilarity(fp, pf))
    return max(sims) if sims else 0.0

def accuracy_at_k(true_mol, pred_mols, k):
    if true_mol is None or not pred_mols or k <= 0:
        return 0
    true_block = inchi_block(true_mol)
    if true_block is None:
        return 0
    topk = pred_mols[:k]
    return int(any((pm is not None) and (inchi_block(pm) == true_block) for pm in topk))

def mces_min_at_k(mces_fn, true_smiles, pred_smiles, k, cap):
    vals = []
    for smi in pred_smiles[:k]:
        if smi is None:
            vals.append(cap)
        else:
            vals.append(mces_fn(true_smiles, smi))
    return min(vals) if vals else cap


# ------------------------- plotting ------------------------------------------
def _ensure_matplotlib_headless():
    # ensure plotting works on servers without a display
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        pass
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt

def make_histograms(per_df, ks, out_dir, bins=40, dpi=200):
    _, plt = _ensure_matplotlib_headless()
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def _plot_one_prefix(prefix, xlabel, fname):
        # per-k histograms + gray "sum of distributions" overlay
        series_by_k = []
        for k in ks:
            col = f"{prefix}@{k}"
            if col in per_df.columns:
                series_by_k.append((k, per_df[col].dropna().values))
        if not series_by_k:
            return

        # common bins over pooled data
        pooled = np.concatenate([v for _, v in series_by_k]) if series_by_k else np.array([])
        if pooled.size == 0:
            return
        counts_by_k = []
        hist_bins = bins
        # compute bin edges once
        counts0, edges = np.histogram(pooled, bins=hist_bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        # recompute each k with same edges
        for k, arr in series_by_k:
            c, _ = np.histogram(arr, bins=edges)
            counts_by_k.append((k, c))
        # gray sum-of-distributions
        summed = np.sum([c for _, c in counts_by_k], axis=0)

        plt.figure(figsize=(7, 5))
        # background gray bars (sum)
        plt.bar(bin_centers, summed, width=(edges[1]-edges[0]), color="lightgray", edgecolor=None, alpha=0.6, label="sum over k")
        # per-k step lines on top
        for k, c in counts_by_k:
            plt.plot(bin_centers, c, lw=2, label=f"k={k}")
        plt.xlabel(xlabel)
        plt.ylabel("count (samples)")
        plt.title(f"{prefix}: distribution across samples")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, fname), dpi=dpi, bbox_inches="tight")
        plt.close()

        # also save individual per-k histograms
        for k, arr in series_by_k:
            plt.figure(figsize=(7, 5))
            plt.hist(arr, bins=edges, color=None, edgecolor="black")
            plt.xlabel(xlabel)
            plt.ylabel("count (samples)")
            plt.title(f"{prefix} @ k={k}")
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{prefix}_k{k}.png"), dpi=dpi, bbox_inches="tight")
            plt.close()

    # what we can plot
    _plot_one_prefix("tanimoto", "Tanimoto (best in top-k)", "tanimoto_multi_k.png")
    _plot_one_prefix("frac_scaffold", "fraction with scaffold (top-k)", "scaffold_frac_multi_k.png")
    if any(f"acc@{k}" in per_df.columns for k in ks):
        _plot_one_prefix("acc", "accuracy (0/1) in top-k", "accuracy_multi_k.png")
    if any(f"mces_min@{k}" in per_df.columns for k in ks):
        _plot_one_prefix("mces_min", "MCES min (lower is better)", "mces_min_multi_k.png")


# ------------------------- main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_path", required=True, help="CSV with columns: scaffold,pred,true,score,...")
    ap.add_argument("--out_dir", required=True, help="Where to write outputs (CSVs, plots)")
    ap.add_argument("--ks", default="1,10,100", help="Comma-separated list of k values (e.g. '1,10,100,1000')")
    # ↓↓↓ group by TRUE only
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--group_cols", default="true",
                    help="Columns to group predictions by (default: true)")
    ap.add_argument("--enable_mces", action="store_true", help="Compute MCES (off by default)")
    ap.add_argument("--mces_cap", type=float, default=100.0, help="Cap (fallback) for failed MCES")
    ap.add_argument("--mces_threshold", type=float, default=20.0, help="MCES solver threshold")
    ap.add_argument("--make_hists", action="store_true", help="Save histograms to out_dir/plots")
    ap.add_argument("--hist_bins", type=int, default=40, help="Number of bins for histograms")
    ap.add_argument("--hist_dpi", type=int, default=300, help="DPI for histogram PNGs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    df = pd.read_csv(args.file_path)

    required_cols = {"true", "pred", "scaffold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    # ----- Keep ORIGINAL SAMPLE ORDER (by first appearance) -----
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]  # -> ["true"]
    df["_group_id"] = df.groupby(group_cols, sort=False).ngroup()
    # ------------------------------------------------------------

    use_mces = bool(args.enable_mces)

    # Prepare MCES if requested
    my_mces = None
    if use_mces:
        try:
            my_mces = MyopicMCES(threshold=args.mces_threshold)
        except Exception as e:
            raise SystemExit(f"Could not initialize MCES: {e}")

    per_rows = []

    # group by sample (true only)
    for keys, g in tqdm(df.groupby(group_cols, sort=False), desc="Samples"):
        keys = keys if isinstance(keys, tuple) else (keys,)
        sample = {col: val for col, val in zip(group_cols, keys)}

        true_smi = g["true"].iloc[0]
        scaf_smi = g["scaffold"].iloc[0]
        preds_smi = list(g["pred"].astype(str).values)

        # build mols and filter invalid predictions; compute k_eff
        true_mol = mol_from_smiles(true_smi)
        scaf_mol = mol_from_smiles(scaf_smi)

        pred_mols_all = [mol_from_smiles(s) for s in preds_smi]
        valid_pairs = [(s, m) for s, m in zip(preds_smi, pred_mols_all) if m is not None]
        n_valid = len(valid_pairs)

        if n_valid:
            valid_smiles = [s for s, _ in valid_pairs]
            freq = pd.Series(valid_smiles).value_counts()
            
            first_idx = {}
            for i, s in enumerate(preds_smi):  # original order tie-break
                if s not in first_idx:
                    first_idx[s] = i
            valid_smiles_sorted = sorted(valid_smiles, key=lambda x: (-freq[x], first_idx[x]))
            pred_mols_sorted = [mol_from_smiles(s) for s in valid_smiles_sorted]
        else:
            valid_smiles_sorted = []
            pred_mols_sorted = []

        frac_scaf_all = 0.0
        if n_valid:
            scaf_flags_all = [has_scaffold(m, scaf_mol) for _, m in valid_pairs]
            frac_scaf_all = float(sum(scaf_flags_all)) / n_valid

        row = {
            **sample,
            "n_preds": len(preds_smi),
            "n_valid": n_valid,
            "frac_scaffold_all": frac_scaf_all,
        }

        # per-k metrics using k_eff = min(k, n_valid)
        for k in ks:
            k_eff = min(k, n_valid)
            if k_eff > 0:
                top_mols = pred_mols_sorted[:k_eff]
                top_smiles = valid_smiles_sorted[:k_eff]

                flags_k = [has_scaffold(m, scaf_mol) for m in top_mols]
                row[f"frac_scaffold@{k}"] = float(sum(flags_k)) / k_eff

                row[f"acc@{k}"] = accuracy_at_k(true_mol, top_mols, k_eff)
                row[f"tanimoto@{k}"] = tanimoto_max(true_mol, top_mols) if true_mol else 0.0

                if use_mces:
                    row[f"mces_min@{k}"] = mces_min_at_k(my_mces, true_smi, top_smiles, k_eff, cap=float(args.mces_cap))
            else:
                row[f"frac_scaffold@{k}"] = 0.0
                row[f"acc@{k}"] = 0
                row[f"tanimoto@{k}"] = 0.0
                if use_mces:
                    row[f"mces_min@{k}"] = float(args.mces_cap)

        per_rows.append(row)

    # ---------------- save per-sample ----------------
    per_df = pd.DataFrame(per_rows)

    # restore original sample order using _group_id
    order = df[group_cols + ["_group_id"]].drop_duplicates()
    per_df = (per_df
              .merge(order, on=group_cols, how="left")
              .sort_values("_group_id")
              .drop(columns="_group_id"))

    per_csv = os.path.join(args.out_dir, "per_sample_metrics.csv")
    #per_df.to_csv(per_csv, index=False)

    # ---------------- save summaries ----------------
    num_cols = [c for c in per_df.columns if c not in group_cols]
    summary_df = pd.DataFrame({
        "mean_over_samples": per_df[num_cols].mean(numeric_only=True),
        "median_over_samples": per_df[num_cols].median(numeric_only=True)
    }).T
    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    #summary_df.to_csv(summary_csv)

    # compact by-k summary (means only)
    rows = []
    for k in ks:
        row = {"k": k}
        for prefix in ["acc", "tanimoto", "frac_scaffold"]:
            col = f"{prefix}@{k}"
            if col in per_df.columns:
                row[prefix] = float(per_df[col].mean())
        if "mces_min@{}".format(k) in per_df.columns:
            row["mces_min"] = float(per_df[f"mces_min@{k}"].mean())
        rows.append(row)
    byk_df = pd.DataFrame(rows)
    byk_csv = os.path.join(args.out_dir, f"summary_by_k_{args.seed}.csv")
    byk_df.to_csv(byk_csv, index=False)

    # ---------------- plots (optional) -------------
    if args.make_hists:
        make_histograms(per_df, ks, args.out_dir, bins=args.hist_bins, dpi=args.hist_dpi)

    print(f"Wrote per-sample metrics  -> {per_csv}")
    print(f"Wrote summary metrics     -> {summary_csv}")
    print(f"Wrote by-k summary        -> {byk_csv}")
    if args.make_hists:
        print(f"Saved histograms in       -> {os.path.join(args.out_dir, 'plots')}")


if __name__ == "__main__":
    main()