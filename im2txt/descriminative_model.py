
import tensorflow as tf

class DescriminativeModel(object):

  def __init__(self, config):

    # mode 
    self.mode = "train"

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

    # A float32 Tensor with shape [batch_size, embedding_size]
    self.real_vectors = None

    # A float32 Tensor with shape [batch_size, embedding_size]
    self.fake_vectors = None

    # Model configuration
    self.config = config

    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, embedding_size]
    self.image_embeddings = None

    self.var_list = []

    # Loss
    self.G_loss = None
    self.D_loss = None

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
    #self.inception_output, self.real_seqs, self.real_lens, \
    #    self.fake_seqs, self.fake_lens = input_list

    # Delete <S> and </S> in a sentence

    self.inception_output = input_list[0]
    self.real_seqs = input_list[1][:,1:]
    self.real_lens = input_list[2] - 2
    self.fake_seqs = input_list[3][:,1:,:]
    self.fake_lens = input_list[4] - 2

    tf.summary.scalar("real_lens", tf.reduce_mean(self.real_lens))
    tf.summary.scalar("fake_lens", tf.reduce_mean(self.fake_lens))

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
        fake_embeddings = tf.reshape(fake_embeddings, [-1, 19, self.config.embedding_size])

    self.real_embeddings = real_embeddings
    self.fake_embeddings = fake_embeddings

    tf.summary.scalar("real_embeddings", tf.reduce_sum(real_embeddings))
    tf.summary.scalar("fake_embeddings", tf.reduce_sum(fake_embeddings))

  @staticmethod
  def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """
    ind = tf.to_int32(ind)
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

  def build_sentence_embeddings(self):
    """Map seqs into sentences embedding
    
    Args:
      self.real_embeddings
      self.fake_embeddings

    Outputs:
      self.real_vectors
      self.fake_vectors
    """
    with tf.variable_scope("D_") as scope:
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)

      if self.mode == "train":
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)
      with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
        zero_state = lstm_cell.zero_state(
            batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)

        zeros = tf.zeros(self.image_embeddings.get_shape())
        _, initial_state = lstm_cell(zeros, zero_state)

        # Allow lstm variables to be reused
        lstm_scope.reuse_variables()

        real_outputs, _ = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=self.real_embeddings,
            sequence_length=self.real_lens,
            initial_state=initial_state,
            dtype=tf.float32,
            scope=lstm_scope)

        fake_outputs, _ = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=self.fake_embeddings,
            sequence_length=self.fake_lens,
            initial_state=initial_state,
            dtype=tf.float32,
            scope=lstm_scope)

    real_outputs = self.extract_axis_1(real_outputs, self.real_lens - 1)
    fake_outputs = self.extract_axis_1(fake_outputs, self.fake_lens - 1)

    self.real_vectors = real_outputs
    self.fake_vectors = fake_outputs

  @staticmethod
  def right_shift(a):
    #suppose a is a Tensor with shape[bath_size, embedding_size]
    b = []
    for i in range(a.get_shape()[0]):
      b.append(a[i,:])
    b = tf.stack(b)
    return b

  def build_model(self):
    """Build model, compute losses both for G and D

    Args:
      self.image_embeddings
      self.real_vectors
      self.fake_vectors

    Outputs:
      self.G_loss
      self.D_loss
    """
    image_embeddings = self.image_embeddings
    real_vectors = self.real_vectors
    fake_vectors = self.fake_vectors

    tf.summary.scalar("real_vectors_reduce_sum", tf.reduce_sum(real_vectors))
    tf.summary.scalar("fake_vectors_reduce_sum", tf.reduce_sum(fake_vectors))
    tf.summary.scalar("image_embeddings_reduce_sum", tf.reduce_sum(image_embeddings))

    shit_vectors = self.right_shift(real_vectors)

    def related_degree(a, b):
      return tf.sigmoid(tf.reduce_sum(tf.multiply(a, b), 1))

    def dist(a, b):
      return tf.reduce_sum(tf.abs(tf.subtract(a, b)), 1)

    r_xr = related_degree(image_embeddings, real_vectors)
    r_xf = related_degree(image_embeddings, fake_vectors)
    r_xs = related_degree(image_embeddings, shit_vectors)

    tf.summary.scalar("r_real", tf.reduce_mean(r_xr))
    tf.summary.scalar("r_fake", tf.reduce_mean(r_xf))

    # Distance between fake and real using L2 norm.
    d_rf = tf.reduce_sum(tf.square(tf.subtract(real_vectors, fake_vectors)), 1)

    self.G_loss = -tf.log(r_xf) + d_rf
    self.D_loss = -tf.log(r_xr) - tf.log(1 - r_xf) - tf.log(1 - r_xs)

    '''
    d_xr = dist(image_embeddings, real_vectors)
    d_xf = dist(image_embeddings, fake_vectors)
    d_xs = dist(image_embeddings, shit_vectors)

    # Using L2-norm distance here
    d_rf = tf.reduce_sum(tf.square(tf.subtract(real_vectors, fake_vectors)), 1)
    self.G_loss = -d_xf - d_rf
    self.D_loss = -d_xr - tf.log(1 - tf.exp(d_xf)) - tf.log(1 - tf.exp(d_xs))
    '''
    

  def build_var_list(self):
    self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="D_")

  def build(self, input_list):
    self.build_inputs(input_list)
    self.build_image_embedding()
    self.build_seqs_embeddings()
    self.build_sentence_embeddings()
    self.build_model()
    self.build_var_list()