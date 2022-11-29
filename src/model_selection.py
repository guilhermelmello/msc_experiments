from collections import defaultdict
from collections.abc import Iterable
from tensorflow import keras
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

from .dataset import load_experiment_dataset
from .dataset.tokenization import to_tf_dataset
from .dataset.tokenization import tokenize_dataset
from .metrics import BinaryAccuracy
from .metrics import SparseF1Score

import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa


def build_model(
    model_id,
    num_outputs,
    learning_rate,
    metrics,
    loss,
):
    if not isinstance(metrics, Iterable):
        metrics = [metrics]

    try:
        model = (
            TFAutoModelForSequenceClassification
            .from_pretrained(model_id, num_labels=num_outputs)
        )
    except Exception:
        print("Building model: ignoring mismatched sizes.")
        model = (
            TFAutoModelForSequenceClassification
            .from_pretrained(
                model_id, num_labels=num_outputs,
                ignore_mismatched_sizes=True)
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


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

    model = build_model(
        model_id=model_id,
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        metrics=model_metrics,
        loss=model_loss
    )
    return model


def build_regression_model(
    model_id,
    learning_rate=5e-5,
    extra_metrics=list(),
    extra_loss=list()
):
    """
    """
    model_loss = [
        tf.keras.losses.MeanAbsoluteError(name='mae')]
    model_metrics = [
        tf.keras.metrics.MeanSquaredError(name='mse'),
        tfa.metrics.RSquare()
    ]

    model_loss += extra_loss
    model_metrics += extra_metrics

    model = build_model(
        num_outputs=1,
        model_id=model_id,
        learning_rate=learning_rate,
        metrics=model_metrics,
        loss=model_loss
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
        _lr = lr
        if isinstance(lr, tuple):
            assert len(lr) == 2
            # using linear decay
            steps_per_epoch = len(train_dataset)    # number of batches
            num_train_steps = steps_per_epoch * epochs
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=lr[0],
                decay_steps=num_train_steps,
                end_learning_rate=lr[1]
            )

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
                metrics_log[metric][str(_lr)].append(results)

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
        squeeze=False,
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


def _run_model_selection(
    dataset,
    dataset_id,
    model_ids,
    build_model_fn,
    build_model_kwargs=dict(),
    dataset_text_pair=False,
    dataset_train_split='train',
    dataset_validation_split='validation',
    train_batch_size=32,
    train_epochs=10,
    train_execs_pre_trial=5,
    train_lr_values=[1e-5],
    save_dir=None,
):
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
            build_model_fn,
            train_dataset=train_dataset,
            validation_dataset=dev_dataset,
            epochs=train_epochs,
            lr_parameters=train_lr_values,
            executions_per_trial=train_execs_pre_trial,

            # model builder parameters:
            model_id=model_id,
            **build_model_kwargs,
        )

        if save_dir is not None:
            _dataset_id = dataset_id.replace('/', '_')
            _model_id = model_id.replace('/', '_')

            _save_dir = os.path.join(save_dir, _dataset_id)
            if not os.path.exists(_save_dir):
                print('Creating directories:', _save_dir)
                os.makedirs(_save_dir)

            _save_dir = os.path.join(_save_dir, _model_id)
            print('Saving results at:', _save_dir)

        title = f"Model: {model_id}\n"
        title += f"Dataset: {dataset_id}\n"
        title += f"Batch Size: {train_batch_size}"

        save_metrics_log(
            _logs,
            savefig=save_dir,
            title=title
        )

        del tokenizer
        del tokenized_data
        del train_dataset, dev_dataset
        tf.keras.backend.clear_session()
        gc.collect()


def run_classification_model_selection(
    dataset_id, score_threshold=0.5, load_dataset_kwargs=dict(), **kwargs
):
    """
    args:
        dataset_id,
        # dataset,                      -> definidos nessa função
        # build_model_fn,               -> definidos nessa função
        # build_model_kwargs=dict()     -> definidos nessa função
    kwargs:
        model_ids,
        dataset_text_pair=False,
        dataset_train_split='train',
        dataset_validation_split='validation',
        train_batch_size=32,
        train_epochs=10,
        train_execs_pre_trial=5,
        train_lr_values=[1e-5],
        save_dir=None,
    """
    # load dataset
    dataset = load_experiment_dataset(dataset_id, **load_dataset_kwargs)
    _num_classes = dataset['train'].features['label'].num_classes
    _num_outputs = _num_classes if _num_classes > 2 else 1

    # f1 score
    if _num_outputs == 1:
        f1_score = SparseF1Score(
            num_classes=1,
            threshold=score_threshold,
            from_logits=True)
    else:
        f1_score = SparseF1Score(
            num_classes=_num_outputs,
            threshold=None,
            average='weighted')

    _run_model_selection(
        dataset=dataset,
        dataset_id=dataset_id,
        build_model_fn=build_classification_model,
        build_model_kwargs=dict(
            # model_id          -> passado em _run_model_selection
            # learning_rate     -> passado em _run_model_selection
            num_outputs=_num_outputs,
            extra_metrics=[f1_score],
            # extra_loss=list()
        ),
        **kwargs
    )


def run_regression_model_selection(
    dataset_id, load_dataset_kwargs=dict(), **kwargs
):
    """
    args:
        dataset_id,
        # dataset,                      -> definidos nessa função
        # build_model_fn,               -> definidos nessa função
        # build_model_kwargs=dict()     -> definidos nessa função
    kwargs:
        model_ids,
        dataset_text_pair=False,
        dataset_train_split='train',
        dataset_validation_split='validation',
        train_batch_size=32,
        train_epochs=10,
        train_execs_pre_trial=5,
        train_lr_values=[1e-5],
        save_dir=None,
    """
    # load dataset
    dataset = load_experiment_dataset(dataset_id, **load_dataset_kwargs)

    _run_model_selection(
        dataset=dataset,
        dataset_id=dataset_id,
        build_model_fn=build_regression_model,
        build_model_kwargs=dict(
            # model_id          -> passado em _run_model_selection
            # learning_rate     -> passado em _run_model_selection
            # extra_metrics -> pode ser definido
            # extra_loss    -> pode ser definido
        ),
        **kwargs
    )
