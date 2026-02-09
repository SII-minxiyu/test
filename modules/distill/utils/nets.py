def get_network(name: str, **kwargs):
    """
    Factory function to get molecular neural networks from DIG (Deep Graph Learning) library
    
    Args:
        name (str): Name of the network model, can be 'schnet' or 'spherenet'
        **kwargs: Keyword arguments for network initialization
                 SchNet params:
                    - hidden_channels (int): Hidden embedding size
                    - num_filters (int): Number of filters
                    - num_interactions (int): Number of interaction layers
                    - num_gaussians (int): Number of gaussians 
                    - cutoff (float): Cutoff distance for interations
                    - readout (str): Readout function ('mean', 'sum', etc.)
                 SphereNet params:
                    - energy_and_force (bool): Whether to predict energy and force together
                    - cutoff (float): Cutoff distance for interactions
                    - num_layers (int): Number of interaction layers
                    - hidden_channels (int): Hidden embedding size
                    - out_channels (int): Size of output features
                    - int_emb_size (int): Size of interaction embedding
                    - basis_emb_size_dist (int): Size of distance embedding
                    - basis_emb_size_angle (int): Size of angle embedding
                    - basis_emb_size_torsion (int): Size of torsion embedding
                    - outputs_channels (int): Number of output channels
    
    Returns:
        model: Neural network model (SchNet or SphereNet)
    
    Examples:
        >>> schnet = get_network('schnet', hidden_channels=128, num_filters=128)
        >>> spherenet = get_network('spherenet', hidden_channels=128, out_channels=1)
        >>> painnet = get_network('painnet', hidden_channels=128)
    """
    
    # Import required models
    from torch_geometric.nn import SchNet
    from dig.threedgraph.method.spherenet import SphereNet
    from modules.distill.utils.PaiNN import PainnModel
    
    # Convert network name to lowercase for comparison
    name = name.lower()
    
    # Check if network type is valid
    valid_networks = ['schnet', 'spherenet','painnet']
    if name not in valid_networks:
        raise ValueError(f"Network must be one of {valid_networks}")
    
    # try:
    if name == 'schnet':
        # Set default parameters for SchNet if not provided
        default_params = {
            'hidden_channels': 256,
            'num_filters': 128,
            'cutoff': 5.0,
            # 'energy_and_force': False,
            'num_interactions': 3, 
            'num_gaussians': 50
        }
        net_cfgs = kwargs.copy()
        # Update default parameters with provided kwargs
        default_params.update(net_cfgs)
        
        return SchNet(**default_params)
        
    elif name == 'spherenet':
        # Set default parameters for SphereNet if not provided
        default_params = {
            # 'energy_and_force': False,
            'cutoff': 5.0,
            'num_layers': 4,
            'hidden_channels': 128,
            'out_channels': 1,
            'int_emb_size': 64,
            'basis_emb_size_dist': 8,
            'basis_emb_size_angle': 8,
            'basis_emb_size_torsion': 8,
            'out_emb_channels': 256,
            'num_spherical': 3,
            'num_radial': 6,
            'envelope_exponent': 5,
            'num_before_skip': 1,
            'num_after_skip': 2,
            'num_output_layers': 3,
            'output_init': 'GlorotOrthogonal',
            'use_node_features': True
        }
        
        # Update default parameters with provided kwargs
        default_params.update(kwargs)
        
        # Initialize SphereNet model
        return SphereNet(**default_params)
    elif name == 'painnet':
        default_params = {
            'cutoff': 5.0,
            'num_interactions': 3,
            'hidden_state_size': 128,
            'pdb': False
        }
        net_cfgs = kwargs.copy()
        # Update default parameters with provided kwargs
        default_params.update(net_cfgs)
        return PainnModel(**default_params)
