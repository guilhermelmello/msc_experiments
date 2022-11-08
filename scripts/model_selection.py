"""Script to select hyperparameter for model finetuning.


"""
from src.model_selection import run_classification_model_selection

# SCRIPT CONFIGURATION
run_classification_model_selection(
    dataset_id='assin2',
    dataset_text_pair=True,
    dataset_train_split='train',
    dataset_validation_split='validation',

    train_batch_size=128,
    train_epochs=3,
    train_execs_pre_trial=2,
    train_lr_values=[1e-5, 1e-6],     # [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    model_ids=[
        'bert-base-cased',
        'xlm-roberta-base',
    ],

    save_dir='results/model_selection',
)
