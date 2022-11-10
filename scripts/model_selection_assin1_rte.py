"""
Script to select hyperparameter for model finetuning.
"""

from src.model_selection import run_classification_model_selection


# SCRIPT CONFIGURATION
run_classification_model_selection(
    dataset_id='assin_rte',
    dataset_text_pair=True,
    dataset_train_split='train',
    dataset_validation_split='validation',

    train_batch_size=128,
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-4, 1e-5, 5e-5, 1e-6],

    model_ids=[
        'bert-base-cased',
        'bert-base-uncased',
        'bert-large-cased',
        'bert-large-uncased',
        'bert-large-uncased-whole-word-masking',
        'bert-large-cased-whole-word-masking',
        'roberta-base',
        'roberta-large',
        # 'bert-base-multilingual-cased',
        # 'xlm-mlm-17-1280',
        # 'xlm-mlm-100-1280',
        # 'xlm-roberta-base',
        # 'xlm-roberta-large',
        # 'joeddav/xlm-roberta-large-xnli',
        # 'neuralmind/bert-base-portuguese-cased',
        # 'neuralmind/bert-large-portuguese-cased',
    ],

    save_dir='results/model_selection',
)
