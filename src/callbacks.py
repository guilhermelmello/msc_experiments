from transformers.modeling_tf_outputs import ModelOutput

import evaluate
import tensorflow as tf


class F1ScoreCallback(keras.callbacks.Callback):
    def __init__(
            self,
            train_dataset=None,
            validation_dataset=None,
            average='weighted',
            name='F1',
            threshold=.5,   # for 1 output only
            from_logits=False,
        ):
        self.name = name
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.f1_metric = evaluate.load('f1')
        self.f1_average = average
        self.threshold = threshold
        self.from_logits = from_logits

    def _compute_f1_score(self, dataset):
        for features, targets in dataset:
            outputs = self.model.predict(features, verbose=False)
            if isinstance(outputs, ModelOutput):
                outputs = outputs.logits
                self.from_logits = True

            if self.from_logits:
                if outputs.shape[-1] == 1:
                    outputs = tf.math.sigmoid(outputs)
                    outputs = tf.math.greater_equal(outputs, self.threshold)
                    outputs = tf.reshape(outputs, (-1, ))
                else:
                    outputs = tf.argmax(outputs, axis=-1)
            else:
                if outputs.shape[-1] > 1:
                    outputs = tf.argmax(outputs, axis=-1)
                else:
                    outputs = tf.reshape(outputs, (-1, ))
            
            self.f1_metric.add_batch(
                references=targets,
                predictions=outputs
            )
        
        score = self.f1_metric.compute(average=self.f1_average)
        return score['f1']

    def on_epoch_end(self, epoch, logs):
        score = self._compute_f1_score(self.train_dataset)
        logs[self.name] = score

        if self.validation_dataset is not None:
            score = self._compute_f1_score(self.validation_dataset)
            logs['val_' + self.name] = score
