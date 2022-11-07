import datasets
from .template import TextPairTemplate


def load_assin(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='entailment_judgment'
    )
    data = datasets.load_dataset('assin', 'full', task=task)
    return data


def load_assin2(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='entailment_judgment'
    )
    data = datasets.load_dataset('assin2', 'default', task=task)
    return data