# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-09-10 15:21:11
# @Last modified by:   khai
# @Last Modified time: 2020-10-27 14:50:16

import tensorflow as tf

class AugmentedConv():
	# Fin: number of input channel
	# Fout: number of output channel
	# k: convolution kernel size
	# dk: k-depth
	# dv = v-depth
	# Nh: number of attention heads
	def __init__(self, Fin, Fout, k, dk, dv, Nh, relative, stride=(1,1), padding='same'):
		self.Fin = Fin
		self.Fout = Fout
		self.k = k
		self.dk = dk
		self.dv = dv
		self.Nh = Nh
		self.relative = relative
		self.stride = stride
		self.padding = padding

	def augmented_conv2d(self, X):
		_, H, W, _ = X.shape
		conv_out = tf.layers.conv2d(X, self.Fout - self.dv, self.k, self.stride, self.padding)
		flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(X)
		logits = tf.matmul(flat_q, flat_k, transpose_b=True)
		if self.relative:
			h_rel_logits, w_rel_logits = self.relative_logits(q)
			logits += h_rel_logits
			logits += w_rel_logits
		weights = tf.nn.softmax(logits) # weights on the values (matrix V)
		attn_out = tf.matmul(weights, flat_v)
		attn_out = tf.reshape(attn_out, [-1, self.Nh, H, W, self.dv // self.Nh])
		attn_out = self.combine_heads_2d(attn_out)
		attn_out = tf.layers.conv2d(attn_out, self.dv, (1, 1)) # Project heads MHA(X) = Concat[O1,...,ONh]Wo
		return tf.concat([conv_out, attn_out], axis=3)

	def compute_flat_qkv(self, inputs):
		_, H, W, _ = inputs.shape
		qkv = tf.layers.conv2d(inputs, 2 * self.dk + self.dv, (1, 1))
		q, k, v = tf.split(qkv, [self.dk, self.dk, self.dv], axis=3)
		q = self.split_heads_2d(q)
		k = self.split_heads_2d(k)
		v = self.split_heads_2d(v)
		dkh = self.dk // self.Nh
		dvh = self.dv // self.Nh
		q *= dkh ** -0.5 # scaled dot-product
		flat_q = tf.reshape(q, [-1, self.Nh, H * W, dkh]) # B, Nh, H, W, dh -> B, Nh, H*W, dh
		flat_k = tf.reshape(k, [-1, self.Nh, H * W, dkh])
		flat_v = tf.reshape(v, [-1, self.Nh, H * W, dvh])
		return flat_q, flat_k, flat_v, q, k, v

	def split_heads_2d(self,inputs):
		"""Split channels into multiple heads."""
		B, H, W, d = inputs.get_shape().as_list() # batch, height, width, channels(depth)
		ret_shape = [B, H, W, self.Nh, d // self.Nh]
		split = tf.reshape(inputs, ret_shape)
		return tf.transpose(split, [0, 3, 1, 2, 4]) # B, H, W, Nh, dh -> B, Nh, H, W, dh


	def combine_heads_2d(self, inputs):
		transposed = tf.transpose(inputs, [0, 2, 3, 1, 4]) # B, H, W, Nh, dh
		Nh, channels = transposed.shape[-2:].as_list()
		ret_shape = transposed.shape[:-2].as_list() + [Nh * channels]
		return tf.reshape(transposed, ret_shape) # B, H, W, d

	def relative_logits(self, q):
		"""Compute relative logits."""
		q_shape = q.get_shape().as_list()
		dk = q_shape[-1]
		H = q_shape[2]
		W = q_shape[3]
		key_rel_w = tf.get_variable(
			name='key_rel_w', shape=(2 * W - 1, dk),
			initializer=tf.random_normal_initializer(dk ** -0.5)
		)

		rel_logits_w = self.relative_logits_1d(q,
											   key_rel_w, H, W, [0, 1, 2, 4, 3, 5])
		key_rel_h = tf.get_variable(
			name='key_rel_h', shape=(2 * H - 1, dk),
			initializer=tf.random_normal_initializer(dk ** -0.5)
		)

		rel_logits_h = self.relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),
											   key_rel_h, W, H, [0, 1, 4, 2, 5, 3]
		)
		return rel_logits_h, rel_logits_w


	def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
		"""Compute relative logits along one dimenion."""
		rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
		rel_logits = tf.reshape(rel_logits, [-1, self.Nh*H, W, 2*W-1])
		rel_logits = self.rel_to_abs(rel_logits)
		rel_logits = tf.reshape(rel_logits, [-1, self.Nh, H, W, W])
		rel_logits = tf.expand_dims(rel_logits, axis=3)
		rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
		rel_logits = tf.transpose(rel_logits, transpose_mask)
		rel_logits = tf.reshape(rel_logits, [-1, self.Nh, H*W, H*W])
		return rel_logits

	def rel_to_abs(self, x):
		"""Converts tensor from relative to aboslute indexing."""
		_, Nh, L,_ = x.shape
		col_pad = tf.zeros((tf.shape(x)[0], Nh, L, 1))
		x = tf.concat([x, col_pad], axis=3)
		flat_x = tf.reshape(x, [-1, Nh, L * 2 * L])
		flat_pad = tf.zeros((tf.shape(x)[0], Nh, L - 1))
		flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
		final_x = tf.reshape(flat_x_padded, [-1, Nh, L+1, 2*L-1])
		final_x = final_x[:, :, :L, L-1:]
		return final_x


