from .loaders import *


_dataset_loaders = {
    'assin': load_assin,
    'assin2': load_assin2
}


def load_experiment_dataset(dataset_id, **kwargs):
    if dataset_id not in _dataset_loaders:
        raise ValueError(f"Could not find dataset: {dataset_id}")
    
    print(f"Loadind dataset: {dataset_id}")
    dataset_loader = _dataset_loaders[dataset_id]
    data = dataset_loader(**kwargs)
    return data