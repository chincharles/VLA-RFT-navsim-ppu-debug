"""Lazy public imports allow the standalone HF encoder without robot dataset dependencies."""
def __getattr__(name):
    if name in ('available_model_names','available_models','get_model_description','load'):
        from . import models
        return getattr(models,name)
    raise AttributeError(name)
