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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import configuration
import generative_model

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = "/media/mazm13/mscoco/train-?????-of-00256"
  model_config.mle_checkpoint_file = "./ckpt_file/train/model.ckpt-457157"
  model.config.vocab_file = "/media/mazm13/mscoco/word_counts.txt"
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = generative_model.GenerativeModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()

  with tf.Session(graph=g) as sess:
    model.init_fn(sess)
    for i in xrange(20):
      sentences = tf.sess.run(model.fake_seqs)
      for sentence in sentences:
        nl_sen = [model.vocab.id_to_word(word_id) for word_id in sentence]
        print nl_sen

if __name__ == "__main__":
  tf.app.run()
