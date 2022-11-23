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

    train_batch_size=128,    # base: 128     large: 64
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-4, 5e-5, 1e-5, 1e-6, (1e-5, 1e-7), (1e-5, 0)],

    model_ids=[
        'bert-base-cased',                                      # (  ) - 128
        'bert-base-uncased',                                    # (  ) - 128
        'roberta-base',                                         # (  ) - 128
        'bert-base-multilingual-cased',                         # (  ) - 128
        'neuralmind/bert-base-portuguese-cased',                # (  ) - 128

        'bert-large-cased',                                     # (  ) - 128
        'bert-large-uncased',                                   # (  ) - 128
        'bert-large-uncased-whole-word-masking',                # (  ) - 128
        'bert-large-cased-whole-word-masking',                  # (  ) - 128
        'roberta-large',                                        # (  ) - 128
        'xlm-mlm-17-1280',                                      # (  ) - 128
        'xlm-mlm-100-1280',                                     # (  ) - 128
        'xlm-roberta-base',                                     # (  ) - 128
        'xlm-roberta-large',                                    # (  ) - 128
        'joeddav/xlm-roberta-large-xnli',                       # (  ) - 128
        'tf_models/neuralmind/bert-large-portuguese-cased',     # (  ) - 128
    ],

    save_dir='results/model_selection',
)
