#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# ------------------------ helpers: parsing/sanitizing -------------------------
_WS_RE = re.compile(r"[\u00A0\u2007\u202F]")  # non-breaking spaces

def clean_smiles(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'")
    s = _WS_RE.sub(" ", s).replace(" ", "")
    return s or None

def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    s = clean_smiles(smi)
    if not s:
        return None
    m = Chem.MolFromSmiles(s, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    Chem.rdDepictor.Compute2DCoords(m)
    return m

# ------------------------ MCS + indexing utilities ---------------------------
def find_mcs(m1: Chem.Mol, m2: Chem.Mol,
             ring_matches_ring_only=True,
             complete_rings_only=False,
             timeout_s: int = 10) -> Tuple[Optional[Chem.Mol], Tuple[int, ...], Tuple[int, ...]]:
    res = rdFMCS.FindMCS(
        [m1, m2],
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=complete_rings_only,
        timeout=timeout_s
    )
    if res.canceled or not res.smartsString:
        return None, tuple(), tuple()
    mcs_mol = Chem.MolFromSmarts(res.smartsString)
    if mcs_mol is None:
        return None, tuple(), tuple()
    match1 = m1.GetSubstructMatch(mcs_mol)
    match2 = m2.GetSubstructMatch(mcs_mol)
    if not match1 or not match2:
        return None, tuple(), tuple()
    return mcs_mol, tuple(int(i) for i in match1), tuple(int(i) for i in match2)

def bonds_from_atom_set(mol: Chem.Mol, atom_idx_iter) -> List[int]:
    idx_set = set(int(a) for a in atom_idx_iter)
    keep = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in idx_set and a2 in idx_set:
            keep.append(int(b.GetIdx()))
    return keep

def _as_int_list(x):
    if x is None:
        return []
    out = []
    for v in x:
        if isinstance(v, (list, tuple, set, np.ndarray)):
            out.extend(int(i) for i in v)
        else:
            out.append(int(v))
    return out

def _ensure_list_of_lists(n_mols, items):
    """None -> None; [] -> [[],...]; or broadcast single list to per-mol lists"""
    if items is None:
        return None
    if len(items) == 0:
        return [[] for _ in range(n_mols)]
    if isinstance(items[0], (int, np.integer)):
        # single flat list -> broadcast
        lst = _as_int_list(items)
        return [lst for _ in range(n_mols)]
    # already list-of-lists
    return [ _as_int_list(it) for it in items ]

# ------------------------ drawing primitives ---------------------------------
def draw_grid_png(mols, legends=None, hatoms=None, hbonds=None,
                  cols=2, cell_w=500, cell_h=400, out_png="out.png"):
    if legends is None:
        legends = [""] * len(mols)
    hatoms = _ensure_list_of_lists(len(mols), hatoms)
    hbonds = _ensure_list_of_lists(len(mols), hbonds)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(cell_w, cell_h),
        legends=legends,
        highlightAtomLists=hatoms,
        highlightBondLists=hbonds,
        useSVG=False,   # returns a PIL Image
    )
    img.save(out_png)

def draw_overlay_png(m1, m2, hatoms1=None, hbonds1=None, hatoms2=None, hbonds2=None,
                     w=700, h=500, legend=None, out_png="out.png"):
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)  # modern signature: (width, height)
    d2d.drawOptions().addStereoAnnotation = False

    d2d.DrawMolecule(
        m1,
        highlightAtoms=_as_int_list(hatoms1 or []),
        highlightBonds=_as_int_list(hbonds1 or []),
        legend=legend or ""
    )
    d2d.DrawMolecule(
        m2,
        highlightAtoms=_as_int_list(hatoms2 or []),
        highlightBonds=_as_int_list(hbonds2 or []),
    )
    d2d.FinishDrawing()
    with open(out_png, "wb") as f:
        f.write(d2d.GetDrawingText())

def draw_single_png(m, hatoms=None, hbonds=None, w=600, h=450, legend=None, out_png="out.png"):
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d2d.DrawMolecule(
        m,
        highlightAtoms=_as_int_list(hatoms or []),
        highlightBonds=_as_int_list(hbonds or []),
        legend=legend or "",
    )
    d2d.FinishDrawing()
    with open(out_png, "wb") as f:
        f.write(d2d.GetDrawingText())

# ------------------------ main workflow --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot two molecules and highlight their MCS.")
    ap.add_argument("--data", required=True, help="CSV with columns: identifier, smiles")
    ap.add_argument("--id1", required=True, help="First MSG identifier")
    ap.add_argument("--id2", required=True, help="Second MSG identifier")
    ap.add_argument("--out_prefix", required=True, help="Output path prefix (e.g., ./plots/m1_m2)")
    ap.add_argument("--layout", choices=["side-by-side", "overlay"], default="side-by-side",
                    help="How to plot the two molecules")
    ap.add_argument("--complete_rings_only", action="store_true", help="MCS must contain only complete rings")
    ap.add_argument("--rings_match_rings_only", action="store_true",
                    help="Only match rings to rings in MCS")
    ap.add_argument("--mcs_timeout", type=int, default=10, help="MCS timeout (seconds)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if not {"identifier", "smiles"}.issubset(df.columns):
        raise SystemExit("Input CSV must have columns: identifier, smiles")

    id2smi = dict(zip(df["identifier"].astype(str), df["smiles"].astype(str)))
    for need in (args.id1, args.id2):
        if need not in id2smi:
            raise SystemExit(f"Identifier '{need}' not found in data.")

    m1 = mol_from_smiles(id2smi[args.id1])
    m2 = mol_from_smiles(id2smi[args.id2])
    if m1 is None or m2 is None:
        raise SystemExit("Failed to parse one of the molecules from SMILES.")

    mcs_mol, match1, match2 = find_mcs(
        m1, m2,
        ring_matches_ring_only=args.rings_match_rings_only,
        complete_rings_only=args.complete_rings_only,
        timeout_s=args.mcs_timeout
    )
    if mcs_mol is None:
        print("[warn] MCS not found (timeout or empty); continuing with unhighlighted drawings.")
        match1, match2 = tuple(), tuple()

    mcs_bonds1 = bonds_from_atom_set(m1, match1)
    mcs_bonds2 = bonds_from_atom_set(m2, match2)

    n_mcs_atoms = mcs_mol.GetNumAtoms() if mcs_mol is not None else 0
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    denom = max(n1, n2) if max(n1, n2) > 0 else 1
    ratio = n_mcs_atoms / denom
    print(f"MCS atoms: {n_mcs_atoms} | mol1 atoms: {n1} | mol2 atoms: {n2} | ratio: {ratio:.4f}")

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.layout == "side-by-side":
        out_img = f"{args.out_prefix}_side_by_side.png"
        draw_grid_png(
            [m1, m2],
            legends=[args.id1, args.id2],
            hatoms=[list(match1), list(match2)],
            hbonds=[mcs_bonds1, mcs_bonds2],
            cols=2, cell_w=500, cell_h=400,
            out_png=out_img
        )
    else:
        out_img = f"{args.out_prefix}_overlay.png"
        draw_overlay_png(
            m1, m2,
            hatoms1=list(match1), hbonds1=mcs_bonds1,
            hatoms2=list(match2), hbonds2=mcs_bonds2,
            w=700, h=500, legend=f"{args.id1} vs {args.id2}",
            out_png=out_img
        )
    print(f"[ok] Saved pair image -> {out_img}")

    if mcs_mol is not None and mcs_mol.GetNumAtoms() > 0:
        Chem.rdDepictor.Compute2DCoords(mcs_mol)
        out_mcs = f"{args.out_prefix}_mcs.png"
        draw_single_png(mcs_mol, legend=f"MCS (atoms={n_mcs_atoms})", out_png=out_mcs)
        print(f"[ok] Saved MCS image  -> {out_mcs}")

    out_txt = f"{args.out_prefix}_mcs_ratio.txt"
    with open(out_txt, "w") as f:
        f.write(f"id1={args.id1}\n")
        f.write(f"id2={args.id2}\n")
        f.write(f"mcs_atoms={n_mcs_atoms}\n")
        f.write(f"mol1_atoms={n1}\n")
        f.write(f"mol2_atoms={n2}\n")
        f.write(f"ratio={ratio:.6f}\n")
    print(f"[ok] Wrote MCS stats   -> {out_txt}")

if __name__ == "__main__":
    main()