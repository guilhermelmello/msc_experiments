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
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=self.num_classes, axis=-1)
        if len(y_true.shape) == 3:
            # it seems that a bug exists when building
            # the model with symbolic tensors. 'y_true'
            # is considered a 2D tensor (expanded dim)
            # instead of a 1D tensor.
            y_true = tf.reshape(y_true, (-1, self.num_classes))
        return super().update_state(y_true, y_pred, sample_weight)