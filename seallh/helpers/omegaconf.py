from omegaconf import OmegaConf

def _register():

    # register custom resolvers if not already registered
    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
        OmegaConf.register_new_resolver("div", lambda x, y: x / y)
        OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
        OmegaConf.register_new_resolver("max", lambda *x: max(x))
        OmegaConf.register_new_resolver("min", lambda *x: min(x))

        OmegaConf.register_new_resolver("get_idx", lambda x, idx: x[idx])


_register()
