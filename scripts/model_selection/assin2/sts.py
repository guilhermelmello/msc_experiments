"""
Script to select hyperparameter for model finetuning.
"""

from src.model_selection import run_regression_model_selection


# SCRIPT CONFIGURATION
run_regression_model_selection(
    dataset_id='assin2_sts',
    dataset_text_pair=True,
    dataset_train_split='train',
    dataset_validation_split='validation',

    train_batch_size=64,    # base: 128     large: 64
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-4, 5e-5, 1e-5, 1e-6, (1e-5, 1e-7), (1e-5, 0)],

    model_ids=[
        # 'bert-base-cased',                                      # (OK) - 128
        # 'bert-base-uncased',                                    # (OK) - 128
        # 'roberta-base',                                         # (OK) - 128
        # 'bert-base-multilingual-cased',                         # (OK) - 128
        # 'neuralmind/bert-base-portuguese-cased',                # (OK) - 128

        # 'bert-large-cased',                                     # (OK) - 64
        # 'bert-large-uncased',                                   # (OK) - 64
        # 'bert-large-uncased-whole-word-masking',                # (OK) - 64
        # 'bert-large-cased-whole-word-masking',                  # (OK) - 64
        # 'roberta-large',                                        # (OK) - 64
        # 'xlm-mlm-17-1280',                                      # (OK) - 64
        # 'xlm-mlm-100-1280',                                     # (OK) - 64
        # 'xlm-roberta-base',                                     # (OK) - 64
        # 'xlm-roberta-large',                                    # (OK) - 64
        # 'joeddav/xlm-roberta-large-xnli',                       # (OK) - 64
        # 'tf_models/neuralmind/bert-large-portuguese-cased',     # (OK) - 64
    ],

    save_dir='results/model_selection',
)
