import datasets
from .template import TextTemplate
from .template import TextPairTemplate


def load_assin_rte(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='entailment_judgment'
    )
    data = datasets.load_dataset('assin', 'full', task=task)
    return data


def load_assin_sts(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='relatedness_score')
    data = datasets.load_dataset('assin', 'full', task=task)
    return data


def load_assin2_rte(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='entailment_judgment'
    )
    data = datasets.load_dataset('assin2', 'default', task=task)
    return data


def load_assin2_sts(**kwargs):
    task = TextPairTemplate(
        text_column='premise',
        text_pair_column='hypothesis',
        label_column='relatedness_score'
    )
    data = datasets.load_dataset('assin2', 'default', task=task)
    return data


def load_bpsad_polarity(data_dir, **kwargs):
    task = TextTemplate(
        text_column='review_text',
        label_column='polarity'
    )
    data = datasets.load_dataset(
        path='lm4pt/bpsad',
        name='polarity',
        data_dir=data_dir,
        task=task
    )
    return data


def load_bpsad_rating(data_dir, **kwargs):
    task = TextTemplate(
        text_column='review_text',
        label_column='rating'
    )
    data = datasets.load_dataset(
        path='lm4pt/bpsad',
        name='rating',
        data_dir=data_dir,
        task=task
    )
    return data
