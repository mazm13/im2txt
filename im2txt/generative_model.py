# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops
from inference_utils import vocabulary


class GenerativeModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # Vocabulary
    self.vocab = vocabulary.Vocabulary(self.config.vocab_file)

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # A float32 Tensor with shape [batch_size,]
    self.image_ids = None

    # An int32 Tensor with shape [batch_size, padded_length], for INFERENCE mode.
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.real_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.real_mask = None

    # Generated sentences list for images
    self.fake_seqs = []

    # Lengths of generated sentences
    self.fake_lens = []

    # Generated sentences of word probs. A float32 Tensor with shape 
    #   [batch_size, max_length, vocab_size], i.e. [4, 20, 12000]
    self.fake_seqs_probs = None

    # Lengths Tensor of generated sentences. A int32 Tensor with shape [batch_size,]
    self.fake_lens_tensor = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, inception_v3_output_size], i.e. [4, 2048]
    self.inception_output = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Collection of variables from the MLE generator submodel.
    self.var_list = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No real sequences or real mask in inference mode.
      real_seqs = None
      real_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, image_id, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, image_id, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, image_ids, real_seqs, real_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))
      input_seqs = None

    self.images = images
    self.image_ids = image_ids
    self.real_seqs = real_seqs
    self.real_mask = real_mask
    self.input_seqs = input_seqs

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Save inception output for Descrimitive model
    self.inception_output = inception_output

    # Collect all variables in MLE(Show and Tell)
    self.var_list.extend(self.inception_variables)

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope,
          trainable=False)

    image_embedding_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="image_embedding")
    # Collect all variables in MLE(Show and Tell)
    self.var_list.extend(image_embedding_variables)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

  def build_word_embeddings(self):
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)

    seq_embedding_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="seq_embedding")
    # Collect all variables in MLE(Show and Tell)
    self.var_list.extend(seq_embedding_variables)

  def word_embedding(self, word_id):
    word_id = tf.expand_dims(word_id, 0)
    with tf.variable_scope("seq_embedding", reuse=True), tf.device("/cpu:0"):
      embedding_map = tf.get_variable("map")
      word_embedding_ = tf.nn.embedding_lookup(embedding_map, word_id)
    return word_embedding_

  def word_probs_embedding(self, word_probs):
    # word_probs is a float32 Tensor with shape [vocab_size,]
    # change it to [1, vocab_size]
    word_probs = tf.expand_dims(word_probs, 0)
    with tf.variable_scope("seq_embedding", reuse=True), tf.device("/cpu:0"):
      embedding_map = tf.get_variable("map")
      word_embeddding_ = tf.matmul(word_probs, embedding_map)
    return word_embeddding_

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings

    Outputs:
      self.fake_seqs
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    fake_seqs_probs = []
    # Generate sentences of each image one by one
    for i in range(self.image_embeddings.get_shape()[0]):
      # Fetch ith image embedding of image_embeddings
      image_embedding_ = self.image_embeddings[i,:]
      image_embedding_ = tf.expand_dims(image_embedding_, 0)
      with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
        # Share variables for i > 0
        if i > 0: lstm_scope.reuse_variables()
        # Feed the image_embedding_ to set the initial LSTM state.   
        zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
        _, initial_state = lstm_cell(image_embedding_, zero_state)

      start_id = tf.constant(self.vocab.start_id, dtype=tf.int64)
      sentence = [start_id]
      length = 20

      lstm_inputs = self.word_embedding(start_id)
      lstm_state = initial_state

      sentence_probs = [tf.one_hot(start_id, depth=self.config.vocab_size)]

      for j in range(self.config.max_length - 1):
        with tf.variable_scope("lstm", reuse=True) as lstm_scope:
          lstm_outputs, state_tuple = lstm_cell(
              inputs=lstm_inputs,
              state=lstm_state)
        with tf.variable_scope("logits") as logits_scope:
          if i > 0 or j > 0: logits_scope.reuse_variables()
          logits = tf.contrib.layers.fully_connected(
              inputs=lstm_outputs,
              num_outputs=self.config.vocab_size,
              activation_fn=None,
              weights_initializer=self.initializer,
              scope=logits_scope)
        # Multiply a \Beta before category distribution, \Beta = 3
        logits = tf.nn.softmax(3 * logits[0])

        logits = tf.log(tf.clip_by_value(logits, 1e-10, 1.0))
        # Do gumbel-softmax 
        gumbel_noise = np.random.gumbel(0.0,1.0,size=logits.get_shape())
        gumbel = logits + gumbel_noise
        index = tf.argmax(gumbel)

        word_probs = tf.nn.softmax(gumbel / self.config.gumbel_temperature)
        sentence_probs.append(word_probs)

        sentence.append(index)

        if self.mode == "train":
          lstm_inputs = self.word_probs_embedding(word_probs)
        else:
          lstm_inputs = self.word_embedding(index)

        lstm_state = state_tuple

      # With considering that there is must no if condition in TensorFlow
      # So we find the first end_id in sentence
      # And there may be no end_id in sentence, so we add end_id in sentence
      end_id = tf.constant(self.vocab.end_id, dtype=tf.int64)
      sentence_ = sentence
      sentence_.append(end_id)

      length = tf.reduce_min(tf.where(tf.equal(tf.stack(sentence_), end_id)))
      length = tf.reduce_min([length+1, 20])

      fake_seqs_probs.append(sentence_probs)

      self.fake_lens.append(length)
      self.fake_seqs.append(sentence)
      

    self.fake_seqs_probs = tf.stack(fake_seqs_probs)
    self.fake_lens_tensor = tf.stack(self.fake_lens)

    lstmed_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="lstm")
    logits_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="logits")
    # Collect all variables in MLE(Show and Tell)
    self.var_list.extend(lstmed_variables)
    self.var_list.extend(logits_variables)

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_generator_initializer(self):
    """Sets up the function to restore generator variables from MLE checkpoint."""
    if self.mode != "inference":
      # Restore MLE variables only.
      saver = tf.train.Saver(self.var_list)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.mle_checkpoint_file)
        saver.restore(sess, self.config.mle_checkpoint_file)

      self.init_fn = restore_fn    

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def output(self):
    real_lens = tf.reduce_sum(self.real_mask, 1)
#    return self.inception_output, self.real_seqs, real_lens, \
#        self.fake_seqs_probs, self.fake_lens_tensor
    return self.image_embeddings, self.real_seqs, real_lens, \
        self.fake_seqs_probs, self.fake_lens_tensor
  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_word_embeddings()
    self.build_model()
    self.setup_generator_initializer()
    self.setup_global_step()
