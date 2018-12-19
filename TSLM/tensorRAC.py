import collections
import hashlib
import numbers

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import LSTMStateTuple

class TensorRAC(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, activation=None, reuse=None):
    super(TensorRAC, self).__init__(_reuse=reuse)


    self._num_units = num_units
    self._activation = activation or tf.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

    

  def call(self, inputs, state):
    self._kernel_U = self.add_variable(
        'U',
        shape=[inputs.shape[-1], self._num_units], dtype=tf.float32)
    self._kernel_W = self.add_variable(
        'W',
        shape=[self._num_units, self._num_units], dtype=tf.float32)
    self._kernel_b1 = self.add_variable(
        'b1',
        shape=[self._num_units], dtype=tf.float32)
    self._kernel_b2 =  self.add_variable(
        'b2',
        shape=[self._num_units], dtype=tf.float32)
    a = tf.matmul(inputs, self._kernel_U)+self._kernel_b1
    b = tf.matmul(state, self._kernel_W)+self._kernel_b2 
    # a = tf.matmul(inputs, self._kernel_U)
    # b = tf.matmul(state, self._kernel_W) 
    # output = tf.nn.relu(-tf.log(a)-tf.log(b))
    output = a * b
    return output, output
  
  # def call(self,inputs,state):
  #   with tf.variable_scope('Worker'):
  #     self.Wi_worker = tf.Variable(tf.random_normal([self._num_units, self._num_units], stddev=0.1))
  #     self.Ui_worker = tf.Variable(tf.random_normal([inputs.shape[-1], self._num_units], stddev=0.1))
  #     self.bi_worker = tf.Variable(tf.random_normal([self._num_units], stddev=0.1))

  #     self.Wf_worker = tf.Variable(tf.random_normal([self._num_units, self._num_units], stddev=0.1))
  #     self.Uf_worker = tf.Variable(tf.random_normal([inputs.shape[-1], self._num_units], stddev=0.1))
  #     self.bf_worker = tf.Variable(tf.random_normal([self._num_units], stddev=0.1))

  #     self.Wog_worker = tf.Variable(tf.random_normal([self._num_units, self._num_units], stddev=0.1))
  #     self.Uog_worker = tf.Variable(tf.random_normal([inputs.shape[-1], self._num_units], stddev=0.1))
  #     self.bog_worker = tf.Variable(tf.random_normal([self._num_units], stddev=0.1))

  #     self.Wc_worker = tf.Variable(tf.random_normal([self._num_units, self._num_units], stddev=0.1))
  #     self.Uc_worker = tf.Variable(tf.random_normal([inputs.shape[-1], self._num_units], stddev=0.1))
  #     self.bc_worker = tf.Variable(tf.random_normal([self._num_units], stddev=0.1))
  #     params.extend([
  #       self.Wi_worker, self.Ui_worker, self.bi_worker,
  #       self.Wf_worker, self.Uf_worker, self.bf_worker,
  #       self.Wog_worker, self.Uog_worker, self.bog_worker,
  #       self.Wc_worker, self.Uc_worker, self.bc_worker])

  #     def unit(x, hidden_memory_tm1):
  #       previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

  #       # Input Gate
  #       i = tf.sigmoid(
  #           tf.matmul(x, self.Wi_worker) +
  #           tf.matmul(previous_hidden_state, self.Ui_worker) + self.bi_worker
  #       )

  #       # Forget Gate
  #       f = tf.sigmoid(
  #           tf.matmul(x, self.Wf_worker) +
  #           tf.matmul(previous_hidden_state, self.Uf_worker) + self.bf_worker
  #       )

  #       # Output Gate
  #       o = tf.sigmoid(
  #           tf.matmul(x, self.Wog_worker) +
  #           tf.matmul(previous_hidden_state, self.Uog_worker) + self.bog_worker
  #       )

  #       # New Memory Cell
  #       c_ = tf.nn.tanh(
  #           tf.matmul(x, self.Wc_worker) +
  #           tf.matmul(previous_hidden_state, self.Uc_worker) + self.bc_worker
  #       )

  #       # Final Memory cell
  #       c = f * c_prev + i * c_

  #       # Current Hidden state
  #       current_hidden_state = o * tf.nn.tanh(c)

  #       return current_hidden_state,current_hidden_state
