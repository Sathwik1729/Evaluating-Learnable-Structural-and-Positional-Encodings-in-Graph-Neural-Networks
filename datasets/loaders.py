import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, AddLaplacianEigenvectorPE, Compose

DATASET_NAMES = {
    'cora': 'Cora',
    'citeseer': 'Citeseer',
    'pubmed': 'PubMed',
}


def load_dataset(name, pe_dim=8, data_root='data/', pe_mode='laplacian'):
    """
    Load a Planetoid dataset with optional positional encodings.

    Returns:
        data: PyG Data object with fields:
              data.x          [N, F]    node features (normalized)
              data.y          [N]       class labels
              data.edge_index [2, E]    graph edges
              data.pe         [N, pe_dim]  PE if requested (None if pe_mode='none')
              data.train_mask / val_mask / test_mask
        num_features: int
        num_classes: int
    """
    pyg_name = DATASET_NAMES.get(name.lower())
    if pyg_name is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_NAMES)}")

    transforms = [NormalizeFeatures()]
    if pe_dim > 0 and pe_mode == 'laplacian':
        transforms.append(
            AddLaplacianEigenvectorPE(k=pe_dim, attr_name='pe', is_undirected=True)
        )

    dataset = Planetoid(
        root=data_root,
        name=pyg_name,
        transform=Compose(transforms),
    )
    data = dataset[0]

    if pe_dim == 0 or pe_mode != 'laplacian':
        data.pe = None

    return data, dataset.num_features, dataset.num_classes


def add_rw_pe(data, pe_dim):
    """Compute Random Walk PE: [diag(P^1), ..., diag(P^k)] for each node."""
    data = data.clone()
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    num_nodes = data.num_nodes

    values = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
    adj.setdiag(0)
    adj.eliminate_zeros()

    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_degree = np.zeros_like(degree, dtype=np.float32)
    np.divide(1.0, degree, out=inv_degree, where=degree > 0)
    transition = sp.diags(inv_degree).dot(adj).tocsr()

    power = transition
    pe_columns = []
    for _ in range(pe_dim):
        pe_columns.append(power.diagonal())
        power = power.dot(transition).tocsr()

    pe = np.stack(pe_columns, axis=1).astype(np.float32)
    data.pe = torch.from_numpy(pe)
    return data


def add_random_pe(data, pe_dim):
    """Replace PE with random Gaussian vectors (ablation baseline)."""
    data = data.clone()
    data.pe = torch.randn(data.num_nodes, pe_dim)
    return data


def remove_pe(data):
    """Set PE to None (for standard GNNs that do not use PE)."""
    data = data.clone()
    data.pe = None
    return data
