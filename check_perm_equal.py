from src.data.msgym_dataset import MsGymDataset
from src.data.canopus_dataset import CanopusDataset
from src.data.utils import to_dense
import torch
from src.analysis.rdkit_functions import build_molecule
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm
def create_single_product_molecule(data):
    """Convert a single PyG Data object's product graph to atom_types and edge_types.
    
    Args:
        data: PyG Data object containing p_x, p_edge_index, p_edge_attr
        
    Returns:
        list: [atom_types, edge_types] for the product molecule
    """
    # Create a fake batch index for a single molecule
    batch = torch.zeros(data.p_x.size(0), dtype=torch.long)
    batch_r = torch.zeros(data.x.size(0), dtype=torch.long)
    # Convert to dense representation
    products, p_node_mask = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, batch)
    products = products.mask(p_node_mask, collapse=True)
    reactants, r_node_mask = to_dense(data.x, data.edge_index, data.edge_attr, batch_r)
    reactants = reactants.mask(r_node_mask, collapse=True)
    # Get number of nodes
    n_nodes = data.p_x.size(0)  # For single molecule, it's just the number of nodes
    n_nodes_r = data.x.size(0)
    # Extract atom types and edge types
    atom_types = products.X[0, :n_nodes].cpu()  # First molecule (only one) up to number of nodes
    edge_types = products.E[0, :n_nodes, :n_nodes].cpu()
    atom_types_r = reactants.X[0, :n_nodes_r].cpu()
    edge_types_r = reactants.E[0, :n_nodes_r, :n_nodes_r].cpu()
    
    return [atom_types, edge_types], [atom_types_r, edge_types_r]

# Example usage:
# data = torch.load('./testing.pt')  # Load your single PyG Data object
# product_mol = create_single_product_molecule(data)
# atom_types, edge_types = product_mol


dataset = CanopusDataset(stage='train', root='data/canopus')
# atom_decoder = ['N', 'P', 'B', 'I', 'As', 'Se', 'Cl', 'C', 'F', 'S', 'Br', 'O', 'Si']
atom_decoder = ['Se', 'Si', 'S', 'N', 'I', 'B', 'Br', 'Cl', 'F', 'C', 'O', 'P']

count = 0
for i in tqdm(range(len(dataset))):
    data = dataset[i]
    product_mol, reactants_mol = create_single_product_molecule(data)
    atom_types, edge_types = product_mol
    atom_types_r, edge_types_r = reactants_mol
    product_mol = build_molecule(atom_types, edge_types, atom_decoder)
    reactants_mol = build_molecule(atom_types_r, edge_types_r, atom_decoder)
    Chem.GetSymmSSSR(product_mol)
    Chem.GetSymmSSSR(reactants_mol)
    if '+' in data.r_smiles or '-' in data.r_smiles:
        count += 1
        continue
    assert (data.x == data.p_x).all(), f'{i}: {data.x}, {data.p_x}'
    assert Chem.MolToSmiles(GetScaffoldForMol(product_mol)) == data.p_smiles, f'{i}: {data.p_smiles}, {Chem.MolToSmiles(GetScaffoldForMol(product_mol))}'
    # assert Chem.MolToSmiles(reactants_mol) == data.r_smiles, f'{i}: {data.r_smiles}, {Chem.MolToSmiles(reactants_mol)}'

print(count/len(dataset))