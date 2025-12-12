import importlib


def import_class(dotted_path: str) -> type:
    """Import a class from a dotted path string.
    
    Args:
        dotted_path: Dotted path to the class (e.g., 'torch.nn.CrossEntropyLoss')
        
    Returns:
        The imported class
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class cannot be found in the module
        
    Example:
        >>> cls = import_class('torch.nn.CrossEntropyLoss')
        >>> loss_fn = cls()
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
