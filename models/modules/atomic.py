import tensorflow as tf
import numpy as np
from .attention import point_wise_feed_forward_network

def scaled_weighted_dot_product_attention(q, k, v, weight, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Weighting, note that weight may considered as additive term.
    matmul_qk = matmul_qk

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) + weight

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class AtomicMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(AtomicMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.attention_dense = tf.keras.layers.Dense(1)
        self.distance_dense = tf.keras.layers.Dense(self.num_heads)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, vkq_weight, mask, training=None):
        v, k, q, weight = vkq_weight
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        shift_distance = tf.expand_dims(weight, axis=-1)
        # shift_distance = self.distance_dense(shift_distance)
        shift_distance = tf.matmul(shift_distance, [np.arange(0.1, 10+0.01, 9.9/(self.num_heads-1)).tolist()])
        shift_distance = tf.transpose(shift_distance, perm=[0, 3, 1, 2])
        scaled_attention, attention_weights = scaled_weighted_dot_product_attention(
            q, k, v, shift_distance, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        attention_weights = tf.transpose(attention_weights,
                                        perm=[0, 2, 3, 1])  # (batch_size, seq_len_q, seq_len_k, num_heads)
        """
        with tf.name_scope('test_1'):
            attention_weights = self.attention_dense(attention_weights)  # (batch_size, seq_len_q, seq_len_k, 1)
        with tf.name_scope('test_2'):
            attention_weights = tf.reshape(attention_weights,
                                      (batch_size, -1, attention_weights.shape[2]))  # (batch_size, seq_len_q, seq_len_k)
        """
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output # , attention_weights


class AtomicLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(AtomicLayer, self).__init__()

        self.creator = AtomicMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        # self.updater = AtomicMultiHeadAttention(d_model, num_heads)

    def call(self, state_coeff_weight, mask=None, training=None):
        orbital_state, overlap_weight = state_coeff_weight
        total_state = orbital_state

        # Creator
        # attn_output, new_coeff = self.creator(total_state, total_state, total_state, overlap_weight, mask=mask, training=training)
        attn_output = self.creator([total_state, total_state, total_state, overlap_weight], mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(orbital_state + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        new_state = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return new_state #, new_coeff
