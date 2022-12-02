"""
Script to select hyperparameter for model finetuning.
"""

from src.model_selection import run_classification_model_selection


# SCRIPT CONFIGURATION
run_classification_model_selection(
    dataset_id='bpsad_polarity',
    dataset_text_pair=False,
    dataset_train_split='train',
    dataset_validation_split='validation',
    load_dataset_kwargs={'data_dir': 'datasets/bpsad/'},

    train_batch_size=64,    # base: 64     large: 64
    train_epochs=3,
    train_execs_pre_trial=1,
    train_lr_values=[1e-5],

    model_ids=[
        'bert-base-cased',                                    # (  ) -
        'bert-base-uncased',                                  # (  ) -
        'roberta-base',                                       # (  ) -
        'bert-base-multilingual-cased',                       # (  ) -
        'xlm-roberta-base',                                   # (  ) -
        'neuralmind/bert-base-portuguese-cased',              # (  ) -

        # 'bert-large-cased',                                   # (  ) -
        # 'bert-large-uncased',                                 # (  ) -
        # 'bert-large-uncased-whole-word-masking',              # (  ) -
        # 'bert-large-cased-whole-word-masking',                # (  ) -
        # 'roberta-large',                                      # (  ) -
        # 'xlm-mlm-17-1280',                                    # (  ) -
        # 'xlm-mlm-100-1280',                                   # (  ) -
        # 'xlm-roberta-large',                                  # (  ) -
        # 'joeddav/xlm-roberta-large-xnli',                     # (  ) -
        # 'tf_models/neuralmind/bert-large-portuguese-cased',   # (  ) -
    ],

    save_dir='results/model_selection',
)
