from helper import *
from compgcn_conv import CompGCNConv


class CompGCNTransE(keras.Model):
    def __init__(self, edge_index, edge_type, num_rel, metrics, params):
        super(CompGCNTransE, self).__init__()
        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.drop = tf.keras.layers.Dropout(0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = self.add_weight(shape=(self.p.num_ent, self.p.init_dim), initializer="glorot_normal")
        self.init_rel = self.add_weight(shape=(num_rel, self.p.init_dim), initializer="glorot_normal")
        self.mrr, self.hits1, self.hits3, self.hits10 = metrics
        self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, params=self.p)

    def call(self, inputs, training=False):
        sub, rel = inputs
        self.trainable = training

        r = self.init_rel if self.p.score_func != 'transe' else tf.concat([self.init_rel, -self.init_rel], 0)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)

        x = self.drop(x, training=self.trainable)

        sub_emb = tf.gather(x, sub, axis=0, batch_dims=0)
        rel_emb = tf.gather(r, rel, axis=0, batch_dims=0)
        all_ent = x

        obj_emb = sub_emb + rel_emb

        x = -tf.norm((tf.expand_dims(obj_emb, axis=1)) - all_ent, ord=1, axis=2)
        score = tf.keras.activations.sigmoid(x)

        return score

    @tf.function
    def train_step(self, x, y):
        sub, rel, _ = x
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            pred = self((sub, rel), training=True)
            loss = loss_fn(y, pred)
        trainable_vars = self.trainable_weights
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        return loss

    @tf.function
    def test_step(self, x, y):
        sub, rel, obj = x
        pred = self((sub, rel))
        self.mrr.update_state(y, pred, obj)
        self.hits1.update_state(y, pred, obj)
        self.hits3.update_state(y, pred, obj)
        self.hits10.update_state(y, pred, obj)


class CompGCNDistMult(keras.Model):
    def __init__(self, edge_index, edge_type, num_rel, metrics, params):
        super(CompGCNDistMult, self).__init__()
        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.drop = tf.keras.layers.Dropout(0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.init_embed = self.add_weight(shape=(self.p.num_ent, self.p.init_dim), initializer="glorot_normal")
        self.init_rel = self.add_weight(shape=(num_rel*2, self.p.init_dim), initializer="glorot_normal")

        self.mrr, self.hits1, self.hits3, self.hits10 = metrics

        self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, params=self.p)
        self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, params=self.p)

        self.bias = self.add_weight(shape=(self.p.batch_size, self.p.num_ent), initializer="zeros")

    def call(self, inputs, training=False):
        sub, rel = inputs
        self.trainable = training

        r = self.init_rel
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = self.drop(x, training=self.trainable)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
        x = self.drop(x, training=self.trainable)

        sub_emb = tf.gather(x, sub, axis=0, batch_dims=0)
        rel_emb = tf.gather(r, rel, axis=0, batch_dims=0)
        all_ent = x

        obj_emb = sub_emb * rel_emb

        x = tf.linalg.matmul(obj_emb, tf.transpose(all_ent))
        x += self.bias
        score = tf.keras.activations.sigmoid(x)

        return score

    @tf.function
    def train_step(self, x, y):
        sub, rel, _ = x
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            pred = self((sub, rel), training=True)
            loss = loss_fn(y, pred)
        trainable_vars = self.trainable_weights
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        return loss

    @tf.function
    def test_step(self, x, y):
        sub, rel, obj = x
        pred = self((sub, rel))
        self.mrr.update_state(y, pred, obj)
        self.hits1.update_state(y, pred, obj)
        self.hits3.update_state(y, pred, obj)
        self.hits10.update_state(y, pred, obj)


class CompGCNConvE(keras.Model):
    def __init__(self, edge_index, edge_type, num_rel, metrics, params):
        super(CompGCNConvE, self).__init__()
        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.drop = tf.keras.layers.Dropout(0.3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = self.add_weight(shape=(self.p.num_ent, self.p.init_dim), initializer="glorot_normal")
        self.init_rel = self.add_weight(shape=(num_rel*2, self.p.init_dim), initializer="glorot_normal")

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        self.mrr, self.hits1, self.hits3, self.hits10 = metrics

        self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, params=self.p)
        self.bias = self.add_weight(shape=(self.p.batch_size, self.p.num_ent), initializer="zeros")

        self.bn = tf.keras.layers.BatchNormalization(axis=1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1, scale=False)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        self.hidden_drop = tf.keras.layers.Dropout(0.3)
        self.hidden_drop2 = tf.keras.layers.Dropout(0.3)
        self.feature_drop = tf.keras.layers.Dropout(0.3)
        self.m_conv1 = tf.keras.layers.Conv2D(filters=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                              padding='valid', data_format='channels_first')
        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt

        self.dense = tf.keras.layers.Dense(self.p.embed_dim)

    def call(self, inputs, training=False):
        sub, rel = inputs
        self.trainable = training

        r = self.init_rel if self.p.score_func != 'transe' else tf.concat([self.init_rel, -self.init_rel], 0)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = self.drop(x, training=self.trainable)

        sub_emb = tf.gather(x, sub, axis=0, batch_dims=0)
        rel_emb = tf.gather(r, rel, axis=0, batch_dims=0)
        all_ent = x
        stk_inp = self.concat(sub_emb, rel_emb)

        x1 = self.bn(stk_inp, training=self.trainable)
        x1 = self.m_conv1(x1)
        x1 = self.bn2(x1, training=self.trainable)
        x1 = tf.keras.activations.relu(x1)
        x1 = self.feature_drop(x1, training=self.trainable)
        x1 = tf.reshape(x1, (-1, self.flat_sz))
        x1 = self.dense(x1)
        x1 = self.hidden_drop2(x1, training=self.trainable)
        x1 = self.bn3(x1, training=self.trainable)
        x1 = tf.keras.activations.relu(x1)

        x1 = tf.linalg.matmul(x1, tf.transpose(all_ent))
        x1 += self.bias

        score = tf.keras.activations.sigmoid(x1)
        return score

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = tf.reshape(ent_embed, (-1, 1, self.p.embed_dim))
        rel_embed = tf.reshape(rel_embed, (-1, 1, self.p.embed_dim))

        stack_inp = tf.concat([ent_embed, rel_embed], 1)
        stack_inp = tf.transpose(stack_inp, perm=[0, 2, 1])
        stack_inp = tf.reshape(stack_inp, (-1, 1, 2 * self.p.k_w, self.p.k_h))  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_inp

    @tf.function
    def train_step(self, x, y):
        sub, rel, _ = x
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            pred = self((sub, rel), training=True)
            loss = loss_fn(y, pred)
        trainable_vars = self.trainable_weights
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        return loss

    @tf.function
    def test_step(self, x, y):
        sub, rel, obj = x
        pred = self((sub, rel))
        self.mrr.update_state(y, pred, obj)
        self.hits1.update_state(y, pred, obj)
        self.hits3.update_state(y, pred, obj)
        self.hits10.update_state(y, pred, obj)