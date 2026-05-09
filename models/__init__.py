from .gcn import GCN
from .graphsage import GraphSAGE
from .gat import GAT
from .lspe import LSPE, LegacyLSPE
from .pegat import PEGAT

MODEL_REGISTRY = {
    'gcn': GCN,
    'graphsage': GraphSAGE,
    'gat': GAT,
    'lspe': LSPE,
    'lspe_legacy': LegacyLSPE,
    'pegat': PEGAT,
}


def build_model(config, num_features, num_classes):
    name = config['model'].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not in registry. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        in_channels=num_features,
        hidden_channels=config['hidden_dim'],
        out_channels=num_classes,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        pe_dim=config.get('pe_dim', 8),
        heads=config.get('heads', 4),
        lap_pe_sign_flip=config.get('lap_pe_sign_flip', False),
    )
