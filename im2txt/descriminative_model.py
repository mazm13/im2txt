
import tensorflow as tf

class DescriminativeModel(object):

  def __init__(self, config):

    # A float32 Tensor with shape [batch_size, inception_v3_output_size]
    self.inception_output = None

    # An int32 Tensor with shape [bacth_size, padded_length]
    self.real_seqs = None

    # An int32 Tensor with shape [batch_size,]
    self.real_lens = None

    # A float32 Tensor with shape [batch_size, max_length, vocab_size]
    self.fake_seqs = None

    # An int32 Tensor with shape [batch_size,]
    self.fake_lens = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size]
    self.real_embeddings = None

    # A float32 Tensor with shape [batch_size, max_length, embedding_size]
    self.fake_embeddings = None

    self.config = config

    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, embedding_size]
    self.image_embeddings = None

    self.var_list = []

  def build_inputs(self, input_list):
    """Input prefetching from Generative model

    Args:
      input_list: A list consists of output from G

    Outputs:
      self.inception_output
      self.real_seqs
      self.real_lens
      self.fake_seqs
      self.fake_lens
    """

    assert len(input_list) == 5
    self.inception_output, self.real_seqs, self.real_lens,
        self.fake_seqs, self.fake_lens = input_list

  def build_image_embedding(self):
    """Build Image embedding from inception output

    Args:
      self.inception_output

    Outputs:
      self.image_embeddings
    """

    with tf.variable_scope("D_"):
      with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.inception_output,
            num_outputs=self.config.embedding_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

    self.image_embeddings = image_embeddings

  def build_seqs_embeddings(self):
    """Map word_id into word_imbedding
    
    Args:
      self.real_seqs
      self.fake_seqs
  
    Output：
      self.real_embeddings
      self.fake_embeddings

    """
    with tf.variable_scope("D_"):
      with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name="map",
            shape=[self.config.vocab_size, self.config.embedding_size],
            initializer=self.initializer)
        real_embeddings = tf.nn.embedding_lookup(embedding_map, self.real_seqs)

      with tf.variable_scope("seq_embedding", reuse=True), tf.device("/cpu:0"):
        embedding_map = tf.get_variable("map")
        fake_seqs = tf.reshape(self.fake_seqs, [-1, self.config.vocab_size])
        fake_embeddings = tf.matmul(fake_seqs, embedding_map)
        fake_embeddings = tf.reshape(fake_embeddings, [-1, 20, self.config.embedding_size])

    self.real_embeddings = real_embeddings
    self.fake_embeddings = fake_embeddings

  def build(self, input_list):
    self.build_inputs(input_list)
    self.build_image_embedding()