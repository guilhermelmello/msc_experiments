"""
Script to select hyperparameter for model finetuning.
"""

from src.model_selection import run_classification_model_selection


# SCRIPT CONFIGURATION
run_classification_model_selection(
    dataset_id='assin2_rte',
    dataset_text_pair=True,
    dataset_train_split='train',
    dataset_validation_split='validation',

    train_batch_size=64,    # base: 128     large: 64
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-4, 5e-5, 1e-5, 1e-6, (1e-5, 1e-7)],

    model_ids=[
        'bert-base-cased',                                    # OK
        'bert-base-uncased',                                  # OK
        'bert-large-cased',                                   # OK
        'bert-large-uncased',                                 # OK
        'bert-large-uncased-whole-word-masking',              # OK
        'bert-large-cased-whole-word-masking',                # OK
        'roberta-base',                                       # OK
        'roberta-large',                                      # OK
        'bert-base-multilingual-cased',                       # OK
        'xlm-mlm-17-1280',                                    # OK
        'xlm-mlm-100-1280',                                   # OK
        'xlm-roberta-base',                                   # OK
        'xlm-roberta-large',                                  # OK
        'joeddav/xlm-roberta-large-xnli',                     # OK
        'neuralmind/bert-base-portuguese-cased',              # OK
        'tf_models/neuralmind/bert-large-portuguese-cased',   # OK
    ],

    save_dir='results/model_selection',
)
