from collections import defaultdict
from tensorflow import keras
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

from .dataset import load_experiment_dataset
from .dataset.tokenization import to_tf_dataset
from .dataset.tokenization import tokenize_dataset
from .metrics import BinaryAccuracy
from .metrics import SparseF1Score

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def build_classification_model(
        model_id,
        num_outputs,
        learning_rate=5e-5,
        extra_metrics=list(),
        extra_loss=list()
):
    if num_outputs == 1:
        model_loss = [
            keras.losses.BinaryCrossentropy(from_logits=True)]
        model_metrics = [
            BinaryAccuracy('accuracy', from_logits=True)]
    else:
        model_loss = [
            keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
        model_metrics = [
            tf.metrics.SparseCategoricalAccuracy('accuracy')]

    model_loss += extra_loss
    model_metrics += extra_metrics

    model = (
        TFAutoModelForSequenceClassification
        .from_pretrained(model_id, num_labels=num_outputs)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate),
        loss=model_loss,
        metrics=model_metrics,
    )
    return model


def hyperparameter_search(
        build_model_fn,
        train_dataset,
        validation_dataset,
        epochs=10,
        lr_parameters=0.01,
        executions_per_trial=5,
        callbacks=list(),
        **kwargs
):
    if not isinstance(lr_parameters, list):
        lr_parameters = [lr_parameters]

    # [metric_name][lr_value][execution] -> list(metric result by epoch)
    metrics_log = defaultdict(lambda: defaultdict(list))
    log_msg = "Executing Trial: {}/{} => Model {}/{}"

    total_trials = len(lr_parameters)
    for trial, lr in enumerate(lr_parameters):
        for execution in range(executions_per_trial):
            print(
                log_msg.format(
                    trial+1, total_trials, execution+1, executions_per_trial)
            )
            model = build_model_fn(
                learning_rate=lr,
                **kwargs
            )
            model_hist = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                callbacks=callbacks,
                epochs=epochs
            )

            for metric, results in model_hist.history.items():
                metrics_log[metric][lr].append(results)

    return metrics_log


def save_metrics_log(metrics_log, figsize=None, savefig=None, title=None):
    # compute mean and std by trial
    metric_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for metric_name, metric_results in metrics_log.items():
        for lr_value, lr_results in metric_results.items():
            mean = np.mean(lr_results, axis=0)
            std = np.std(lr_results, axis=0)
            metric_values[metric_name][lr_value]['mean'] = mean
            metric_values[metric_name][lr_value]['std'] = std

    # metric names (not validation names)
    metric_names = [
        m
        for m in metric_values.keys()
        if not m.startswith('val_')]

    # axes grid
    nrows = len(metric_names)
    ncols = len(metric_values[metric_names[0]].keys())
    figsize = figsize if figsize is not None else (5*ncols, 3*nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey='row',
        figsize=figsize
    )

    if title is not None:
        fig.suptitle(title, fontsize=16)

    for row_id, metric in enumerate(metric_names):
        lrs = metric_values[metric].keys()

        for col_id, lr in enumerate(lrs):
            ax = axes[row_id, col_id]

            def _plot_mean_std(ax, name, label):
                mean = metric_values[name][lr]['mean']
                std = metric_values[name][lr]['std']
                x = range(1, len(mean)+1)
                ax.plot(x, mean, label=label)
                ax.fill_between(x, mean+std, mean-std, alpha=.25)

            _plot_mean_std(ax, metric, 'Train')

            val_metric = 'val_' + metric
            if val_metric in metric_values.keys():
                _plot_mean_std(ax, val_metric, 'Validation')

            if row_id == 0:
                ax.set_title(f'Learning Rate: {lr}')
            if row_id == (nrows - 1):
                ax.set_xlabel('Epoch')
            if col_id == 0:
                ax.set_ylabel(metric.capitalize())
            ax.legend()

    if savefig is not None:
        plt.savefig(savefig + '.png')

        json_obj = json.dumps(metrics_log, indent=4)
        with open(savefig + '.json', 'w') as f:
            f.write(json_obj)

    plt.show()


def run_classification_model_selection(
    dataset_id,
    model_ids,
    save_dir,
    dataset_text_pair=False,
    dataset_train_split='train',
    dataset_validation_split='validation',
    train_batch_size=32,
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-5],
):
    # load dataset
    dataset = load_experiment_dataset(dataset_id)
    _num_classes = dataset[dataset_train_split].features['label'].num_classes
    _num_outputs = _num_classes if _num_classes > 2 else 1

    for model_id in model_ids:
        print(f"Searching hyperparameters for: {model_id}")

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # dataset tokenization
        tokenized_data = tokenize_dataset(
            dataset=dataset,
            batch_size=1024,
            tokenizer=tokenizer,
            text_pairs=dataset_text_pair,
        )

        # data preparation
        train_dataset, dev_dataset = to_tf_dataset(
            tokenized_data,
            tokenizer,
            batch_size=train_batch_size,
            train_split=dataset_train_split,
            dev_split=dataset_validation_split
        )

        # model training
        _logs = hyperparameter_search(
            build_classification_model,
            train_dataset,
            dev_dataset,
            epochs=train_epochs,
            lr_parameters=train_lr_values,
            executions_per_trial=train_execs_pre_trial,

            # model builder parameters:
            model_id=model_id,
            num_outputs=_num_outputs,
            extra_metrics=[
                SparseF1Score(
                    num_classes=_num_classes,
                    average='weighted',
                    name='f1_score'),
            ],
        )

        savefig_path = os.path.join(save_dir, dataset_id)
        if not os.path.exists(savefig_path):
            os.makedirs(savefig_path)

        title = f"Model: {model_id}\n"
        title += f"Dataset: {dataset_id}\n"
        title += f"Batch Size: {train_batch_size}"

        save_metrics_log(
            _logs,
            savefig=os.path.join(savefig_path, model_id),
            title=title
        )
