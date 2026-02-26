import argparse
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
from pathlib import Path
import numpy as np

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.msgym_dataset import MsGymDataModule, MsGyminfos
from src.data.canopus_dataset import CanopusDataModule, Canopusinfos
from src.analysis.visualization import MolecularVisualization
from src.frameworks import diffusion_utils
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from pdb import set_trace


def _to_float(x):
    try:
        if hasattr(x, "item"):           # torch/np scalar
            return float(x.item())
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return float(x[0])
        return float(x)
    except Exception:
        return float("nan")

_LF = rdMolStandardize.LargestFragmentChooser()

def main(args):
    set_deterministic(args.sampling_seed)
    torch_device = "cuda:0" if args.device == "gpu" else "cpu"
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(args.sampling_seed if args.sampling_seed is not None else 0)
    diffusion_utils.set_sampling_generator(gen)
    data_root = os.path.join(args.data, args.dataset)
    checkpoint_name = args.checkpoint.split("/")[-1].replace(".ckpt", "")

    output_dir = os.path.join(args.samples, f"{Path(args.msgym_pkl).stem}")
    if args.table_name != '':
        table_name = f"{args.table_name}.csv"
    else: 
        table_name = f"n={args.n_samples}_seed={args.sampling_seed}.csv"
    table_path = os.path.join(output_dir, table_name)

    skip_first_n = 0
    prev_table = pd.DataFrame()
    if os.path.exists(table_path):
        prev_table = pd.read_csv(table_path)
        skip_first_n = len(prev_table) // args.n_samples
        assert len(prev_table) % args.batch_size == 0

    print(f"Skipping first {skip_first_n} data points")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Samples will be saved to {table_path}")

    # Loading model form checkpoint (all hparams will be automatically set)
    if args.model == "Madgen":
        model_class = MarkovBridge
    else:
        raise NotImplementedError(args.model)

    print("Model class:", model_class)

    model = model_class.load_from_checkpoint(args.checkpoint, map_location=torch_device)
    if args.dataset == "msgym":
        datamodule = MsGymDataModule(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            extra_nodes=args.extra_nodes,
            swap=args.swap,
            # added
            msgym_pkl=args.msgym_pkl,
            ranks_pkl=args.ranks_pkl,
            seed=args.sampling_seed 
        )
        dataset_infos = MsGyminfos(datamodule)
    else:
        datamodule = CanopusDataModule(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            extra_nodes=args.extra_nodes,
            swap=args.swap
        )
        dataset_infos = Canopusinfos(datamodule)

    model.eval().to(torch_device)
    visualization_tools = MolecularVisualization(dataset_infos)

    model.visualization_tools = visualization_tools
    model.T = args.n_steps
    group_size = args.n_samples

    ident = 0
    true_molecules_smiles = []
    pred_molecules_smiles = []
    product_molecules_smiles = []
    computed_scores = []
    computed_nlls = []
    computed_ells = []

    dataloader = (
        # datamodule.test_dataloader()[0: int(len(datamodule.test_dataloader())/4)]
        datamodule.test_dataloader()
        if args.mode == "test"
        else datamodule.val_dataloader()
    )
    print("num test batches:", len(dataloader))
    print("dataset length  :", len(dataloader.dataset))
    for i, data in enumerate(tqdm(dataloader)):
        if i * args.batch_size < skip_first_n:
            print(i , skip_first_n)
            continue
        bs = len(data.batch.unique())
        batch_groups = []
        batch_scores = []
        batch_nll = []
        batch_ell = []

        ground_truth = []
        input_products = []
        for sample_idx in range(group_size):
            data = data.to(torch_device)
            (
                pred_molecule_list,
                true_molecule_list,
                products_list,
                scores,
                nlls,
                ells,
            ) = model.sample_batch(
                data=data,
                batch_id=ident,
                batch_size=bs,
                save_final=min(len(dataloader.dataset), args.batch_size),
                keep_chain=min(len(dataloader.dataset), args.batch_size),
                number_chain_steps_to_save=40,
                sample_idx=sample_idx,
                #save_true_reactants=True,
                use_one_hot=args.use_one_hot,
            )

            batch_groups.append(pred_molecule_list)
            batch_scores.append(scores)
            batch_nll.append(nlls)
            batch_ell.append(ells)

            if sample_idx == 0:
                ground_truth.extend(true_molecule_list)
                input_products.extend(products_list)

        # Regrouping sampled reactants for computing top-N accuracy
        grouped_samples = []
        grouped_scores = []
        grouped_nlls = []
        grouped_ells = []
        for mol_idx_in_batch in range(bs):
            mol_samples_group = []
            mol_scores_group = []
            nlls_group = []
            ells_group = []

            for batch_group, scores_group, nll_gr, ell_gr in zip(
                batch_groups, batch_scores, batch_nll, batch_ell
            ):
                mol_samples_group.append(batch_group[mol_idx_in_batch])
                mol_scores_group.append(scores_group[mol_idx_in_batch])
                nlls_group.append(nll_gr[mol_idx_in_batch])
                ells_group.append(ell_gr[mol_idx_in_batch])

            assert len(mol_samples_group) == group_size
            grouped_samples.append(mol_samples_group)
            grouped_scores.append(mol_scores_group)
            grouped_nlls.append(nlls_group)
            grouped_ells.append(ells_group)

        # Writing smiles
        for true_mol, product_mol, pred_mols, pred_scores, nlls, ells in zip(
            ground_truth,
            input_products,
            grouped_samples,
            grouped_scores,
            grouped_nlls,
            grouped_ells
            ):

            # --- TRUE ---
            true_mol = build_molecule(
                true_mol[0], true_mol[1], dataset_infos.atom_decoder
            )
            try:
                true_mol = _LF.choose(true_mol)
            except Exception:
                pass
            true_smi = Chem.MolToSmiles(true_mol, canonical=True)

            # --- SCAFFOLD / PRODUCT ---
            product_mol = build_molecule(
                product_mol[0], product_mol[1], dataset_infos.atom_decoder
            )
            try:
                product_mol = _LF.choose(product_mol)
            except Exception:
                pass
            product_smi = Chem.MolToSmiles(product_mol, canonical=True)

            # --- PREDICTIONS ---
            for pred_mol, pred_score, nll, ell in zip(
                pred_mols, pred_scores, nlls, ells
            ):
                pred_mol, n_dummy_atoms = build_molecule(
                    pred_mol[0],
                    pred_mol[1],
                    dataset_infos.atom_decoder,
                    return_n_dummy_atoms=True,
                )
                try:
                    pred_mol = _LF.choose(pred_mol)
                except Exception:
                    pass
                pred_smi = Chem.MolToSmiles(pred_mol, canonical=True)

                true_molecules_smiles.append(true_smi)
                product_molecules_smiles.append(product_smi)
                pred_molecules_smiles.append(pred_smi)
                computed_scores.append(_to_float(pred_score))
                computed_nlls.append(_to_float(nll))
                computed_ells.append(_to_float(ell))

        table = pd.DataFrame(
            {
                "scaffold": product_molecules_smiles,
                "pred": pred_molecules_smiles,
                "true": true_molecules_smiles,
                "score": computed_scores,
                "nll": computed_nlls,
                "ell": computed_ells,
            }
        )
        full_table = pd.concat([prev_table, table])
        full_table.to_csv(table_path, index=False)
    # Optional: delete cached test file after run
    if args.delete_test_cache:
        root = os.path.join(args.data, args.dataset)
        processed = os.path.join(root, "processed", "msgym_test_final.pt")
        try:
            if os.path.exists(processed):
                os.remove(processed)
                print(f"Deleted {processed}")
        except Exception as e:
            print(f"Could not delete {processed}: {e}")


if __name__ == "__main__":
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=argparse.FileType(mode="r"), required=True)
    parser.add_argument("--checkpoint", action="store", type=str, required=True)
    parser.add_argument("--samples", action="store", type=str, required=True)
    parser.add_argument("--model", action="store", type=str, required=True)
    parser.add_argument("--mode", action="store", type=str, required=True)
    parser.add_argument("--n_samples", action="store", type=int, required=True)
    parser.add_argument(
        "--n_steps", action="store", type=int, required=False, default=None
    )
    parser.add_argument(
        "--sampling_seed", action="store", type=int, required=False, default=None
    )
    parser.add_argument(
        "--use_one_hot", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--table_name", action="store", type=str, required=False, default=''
    )
    # added
    parser.add_argument("--msgym_pkl", type=str, default=None,
                        help="Path to msgym.pkl to use instead of the default.")
    parser.add_argument("--ranks_pkl", type=str, default=None,
                        help="Path to ranks_msgym_pred.pkl (only used for test mode).")
    parser.add_argument("--delete_test_cache", action="store_true",
                    help="Delete processed/msgym_test_final.pt after run.")
    parsed_args, _ = parser.parse_known_args()
    main(args=parse_yaml_config(parsed_args))



