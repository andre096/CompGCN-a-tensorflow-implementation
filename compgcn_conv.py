from helper import *
from message_passing import MessagePassing


def compute_norm(edge_index, num_ent):
    row = edge_index[0]
    col = edge_index[1]
    edge_weight = tf.cast(tf.ones_like(row), tf.float32)

    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_ent)
    deg_inv = tf.pow(deg, -0.5)
    deg_inv = tf.where(tf.math.is_inf(deg_inv), tf.zeros_like(deg_inv),deg_inv)

    norm = tf.gather(deg_inv, row) * edge_weight * tf.gather(deg_inv, col)
    return norm


class CompGCNConv(MessagePassing):
    """
        Base class for Composition Multi-Relational Graph Convolutional Network
    """
    def __init__(self, in_channels, out_channels, num_rel, params=None):
        super(self.__class__, self).__init__()
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rel = num_rel

        # relation-type specific parameter
        self.w_loop = self.add_weight(shape=(in_channels, out_channels), initializer="glorot_normal")
        self.w_in = self.add_weight(shape=(in_channels, out_channels), initializer="glorot_normal")
        self.w_out = self.add_weight(shape=(in_channels, out_channels), initializer="glorot_normal")
        self.w_rel = self.add_weight(shape=(in_channels, out_channels), initializer="glorot_normal")
        self.loop_rel = self.add_weight(shape=(1, in_channels), initializer="glorot_normal")

        self.act = tf.keras.activations.tanh
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.1)
        self.bias = self.add_weight(shape=out_channels, initializer="zeros")

    def call(self, x, edge_index, edge_type, rel_embed):
        """
        :param x: initial tensor entity embedding
        :param edge_index: edge index of adjacency matrix
        :param edge_type: edge type of adjacency matrix
        :param rel_embed: initial tensor relation embedding
        :return: entity and relation embedding tensor after message passing layer

        """

        rel_embed = tf.concat([rel_embed, self.loop_rel], 0)
        num_edges = tf.shape(edge_index)[1]//2
        num_ent = tf.shape(x)[0]

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = tf.stack([tf.range(num_ent), tf.range(num_ent)])
        self.loop_type = tf.fill((num_ent,), tf.shape(rel_embed)[0]-1)

        self.in_norm = compute_norm(self.in_index, num_ent)
        self.out_norm = compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_norm=self.out_norm, mode='out')

        in_res = self.drop(in_res, training=self.trainable)
        out_res = self.drop(out_res, training=self.trainable)

        out = in_res*(1/3) + loop_res*(1/3) + out_res*(1/3)
        out = out + self.bias
        out = self.bn(out, training=self.trainable)
        out = self.act(out)

        return out, tf.linalg.matmul(rel_embed, self.w_rel)[:-1]

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_embed = tf.gather(rel_embed, edge_type)
        xj_rel = self.rel_transform(x_j, rel_embed)
        out = tf.linalg.matmul(xj_rel, weight)

        return out if edge_norm is None else out * tf.reshape(edge_norm,(-1,1))

    def rel_transform(self, ent_embed, rel_embed):
        """
        Composition operation for entities and relations
        """
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rel)
