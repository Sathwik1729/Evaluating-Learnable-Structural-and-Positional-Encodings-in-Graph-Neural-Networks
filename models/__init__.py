from .gcn import GCN

# TODO (Person 2): uncomment these as you implement each model
# from .graphsage import GraphSAGE
# from .gat import GAT
# from .lspe import LSPE

MODEL_REGISTRY = {
    'gcn': GCN,
    # 'graphsage': GraphSAGE,   # uncomment when implemented
    # 'gat': GAT,               # uncomment when implemented
    # 'lspe': LSPE,             # uncomment when implemented
}


def build_model(config, num_features, num_classes):
    """
    Instantiate a model from config dict.

    Config keys used:
      model       (str)   model name, e.g. 'gcn', 'lspe'
      hidden_dim  (int)   hidden layer width
      num_layers  (int)   number of layers
      dropout     (float) dropout rate
      pe_dim      (int)   positional encoding dimension (LSPE only)
      heads       (int)   attention heads (GAT only, default 4)
    """
    name = config['model'].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not in registry. Available: {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[name]
    return model_cls(
        in_channels=num_features,
        hidden_channels=config['hidden_dim'],
        out_channels=num_classes,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        pe_dim=config.get('pe_dim', 8),
        heads=config.get('heads', 4),
    )
