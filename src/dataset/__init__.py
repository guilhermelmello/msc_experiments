from .loaders import *


_dataset_loaders = {
    'assin_rte': load_assin_rte,
    'assin_sts': load_assin_sts,
    'assin2_rte': load_assin2_rte,
    'assin2_sts': load_assin2_sts,
    'bpsad_rating': load_bpsad_rating,
    'bpsad_polarity': load_bpsad_polarity,
}


def load_experiment_dataset(dataset_id, **kwargs):
    if dataset_id not in _dataset_loaders:
        raise ValueError(f"Could not find dataset: {dataset_id}")

    print(f"Loadind dataset: {dataset_id}")
    dataset_loader = _dataset_loaders[dataset_id]
    data = dataset_loader(**kwargs)
    return data
