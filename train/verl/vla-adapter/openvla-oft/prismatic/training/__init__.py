"""Keep training helpers importable without eagerly loading distributed robot trainers."""
def __getattr__(name):
    if name=='get_train_strategy':
        from .materialize import get_train_strategy
        return get_train_strategy
    if name in ('Metrics','VLAMetrics'):
        from . import metrics
        return getattr(metrics,name)
    raise AttributeError(name)
