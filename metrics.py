from helper import *


class MRR(tf.keras.metrics.Metric):
    def __init__(self, name='mrr', dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.mrr = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def reset_state(self):
        self.mrr.assign(0.)
        self.count.assign(0.)

    def result(self):
        return tf.math.divide_no_nan(self.mrr, self.count)

    def update_state(self, y_true, y_pred, obj):
        ranks = self._compute(y_true, y_pred, obj)
        self.mrr.assign_add(tf.reduce_sum(1.0/ranks))
        self.count.assign_add(tf.reduce_sum(tf.ones_like(ranks)))

    def _compute(self, y_true, y_pred, obj):
        b_range = tf.range(tf.shape(y_pred)[0])
        target_pred = tf.gather_nd(y_pred, tf.stack([b_range, obj], axis=-1))  # target_pred=pred[b_range, obj]

        pred = tf.where(tf.cast(y_true, dtype=tf.uint8) != 0, -tf.ones_like(y_pred) * 100000000, y_pred)
        pred = tf.tensor_scatter_nd_update(pred, tf.stack([b_range, obj], axis=-1), target_pred)
        ranks = 1 + tf.argsort((tf.argsort(pred, axis=1, direction='DESCENDING')), axis=-1, direction='ASCENDING')
        ranks = tf.gather_nd(ranks, tf.stack([b_range, obj], axis=-1))
        ranks = tf.cast(ranks, dtype=tf.float32)
        return ranks


class Hits(tf.keras.metrics.Metric):
    def __init__(self, n, name='hits', dtype=None, **kwargs):
        super().__init__('%s@%d' % (name, n), dtype, **kwargs)
        self.n = n
        self.hits = self.add_weight("hits", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def reset_state(self):
        self.hits.assign(0.)
        self.count.assign(0.)

    def result(self):
        return tf.math.divide_no_nan(self.hits, self.count)

    def update_state(self, y_true, y_pred, obj):
        ranks = self._compute(y_true, y_pred, obj)
        self.hits.assign_add(tf.cast(tf.size(ranks[ranks <= self.n]),dtype=tf.float32))
        self.count.assign_add(tf.reduce_sum(tf.ones_like(ranks)))

    def _compute(self, y_true, y_pred, obj):
        b_range = tf.range(tf.shape(y_pred)[0])
        target_pred = tf.gather_nd(y_pred, tf.stack([b_range, obj], axis=-1))  # target_pred=pred[b_range, obj]

        pred = tf.where(tf.cast(y_true, dtype=tf.uint8) != 0, -tf.ones_like(y_pred) * 100000000, y_pred)
        pred = tf.tensor_scatter_nd_update(pred, tf.stack([b_range, obj], axis=-1), target_pred)
        ranks = 1 + tf.argsort((tf.argsort(pred, axis=1, direction='DESCENDING')), axis=-1, direction='ASCENDING')
        ranks = tf.gather_nd(ranks, tf.stack([b_range, obj], axis=-1))
        ranks = tf.cast(ranks, dtype=tf.float32)
        return ranks

