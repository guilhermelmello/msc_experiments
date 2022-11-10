import tensorflow as tf
import tensorflow_addons as tfa


class BinaryAccuracy(tf.metrics.BinaryAccuracy):
    """Add from_logits support."""
    def __init__(self, *args, **kwargs):
        self.from_logits = kwargs.pop('from_logits', False)
        super().__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


class SparseF1Score(tfa.metrics.F1Score):
    """Computes F1 score from tensors with different shapes.

    Usage:
        For models with 1 output:
            num_classes=1, threshold=threshold, from_logits=True/False
        For models with more than 1 output:
            num_classes=N, threshold=None, average='weighted'

    """
    def __init__(self, *args, **kwargs):
        self.from_logits = kwargs.pop('from_logits', False)
        super().__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.num_classes == 1:
            if self.from_logits:
                y_pred = tf.math.sigmoid(y_pred)
            y_true = tf.reshape(y_true, (-1, 1))
        else:
            if len(y_true.shape) == 1:
                y_true = tf.one_hot(y_true, depth=self.num_classes)
            elif y_true.shape[-1] == 1:
                y_true = tf.one_hot(y_true, depth=self.num_classes)
                y_true = tf.reshape(y_true, (-1, self.num_classes))

        return super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        score = super().result()
        if self.average is not None:
            return float(score)
        if self.num_classes == 1:
            return float(score[0])
        return score
