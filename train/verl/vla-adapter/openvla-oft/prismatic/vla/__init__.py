"""Import robot datasets only when explicitly requested."""
def __getattr__(name):
    if name=='get_vla_dataset_and_collator':
        from .materialize import get_vla_dataset_and_collator
        return get_vla_dataset_and_collator
    raise AttributeError(name)
