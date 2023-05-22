import inspect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MessagePassing(keras.layers.Layer):
    """
    Base class for creating message passing layers
    """
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        """
        The initial call to start propagating messages.

        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`) the edge indices,
        and all additional
        data which is needed to construct messages and to update node embeddings.
        """

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tf.shape(tmp)[0]
                message_args.append(tf.gather(tmp, edge_index[0]))
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tf.shape(tmp)[0]
                message_args.append(tf.gather(tmp, edge_index[1]))
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        # message
        out = self.message(*message_args)
        # aggregate
        out = self.aggregate(aggr, out, edge_index[0], size)
        # update
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
        return x_j

    def update(self, aggr_out):
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""
        return aggr_out

    def aggregate(self, name, src, index, dim):
        r"""Aggregates all values from the :attr:`src` tensor at the indices
        specified in the :attr:`index` tensor along the first dimension.
        If multiple indices reference the same location, their contributions
        are aggregated according to :attr:`name` (either :obj:`"add"`,
        :obj:`"mean"` or :obj:`"max"`).

        :param name: The aggregation to use ("add","mean","max").
        :type name: str
        :param src: The source tensor.
        :type src: Tensor
        :param index: The indices of elements to scatter.
        :type index: Tensor
        :param dim: Automatically create output tensor with size.
        :type dim: int
        :return: Return aggregated tensor
        """
        if name == 'add' or name == 'sum':
            out = tf.math.unsorted_segment_sum(src, index, dim)
        elif name == 'max':
            out = tf.math.unsorted_segment_max(src, index, dim)
        else:
            out = tf.math.unsorted_segment_mean(src, index, dim)
        return out
