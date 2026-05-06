import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, AddLaplacianEigenvectorPE, Compose

DATASET_NAMES = {
    'cora': 'Cora',
    'citeseer': 'Citeseer',
    'pubmed': 'PubMed',
}


def load_dataset(name, pe_dim=8, data_root='data/'):
    """
    Load a Planetoid dataset with Laplacian positional encodings.

    Returns:
        data: PyG Data object with fields:
              data.x          [N, F]    node features (normalized)
              data.y          [N]       class labels
              data.edge_index [2, E]    graph edges
              data.pe         [N, pe_dim]  Laplacian eigenvector PE (if pe_dim > 0)
              data.train_mask / val_mask / test_mask
        num_features: int
        num_classes: int
    """
    pyg_name = DATASET_NAMES.get(name.lower())
    if pyg_name is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_NAMES)}")

    transforms = [NormalizeFeatures()]
    if pe_dim > 0:
        transforms.append(
            AddLaplacianEigenvectorPE(k=pe_dim, attr_name='pe', is_undirected=True)
        )

    dataset = Planetoid(
        root=data_root,
        name=pyg_name,
        transform=Compose(transforms),
    )
    data = dataset[0]

    if pe_dim == 0:
        data.pe = None

    return data, dataset.num_features, dataset.num_classes


def add_random_pe(data, pe_dim):
    """Replace Laplacian PE with random vectors (for ablation Experiment 2)."""
    data = data.clone()
    data.pe = torch.randn(data.num_nodes, pe_dim)
    return data


def remove_pe(data):
    """Set PE to None (for running standard GNNs that ignore PE)."""
    data = data.clone()
    data.pe = None
    return data
