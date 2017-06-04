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
  model_config.vocab_file = "/media/mazm13/mscoco/word_counts.txt"
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = "./model/train/"
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Create training directory.
  log_dir = "./model/log/"
  if not tf.gfile.IsDirectory(log_dir):
    tf.logging.info("Creating logging directory: %s", log_dir)
    tf.gfile.MakeDirs(log_dir)  

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = generative_model.GenerativeModel(
        model_config, mode="train", train_inception=False)
    model.build()

    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
    merge = tf.summary.merge_all()

  with tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    model.init_fn(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    for i in xrange(20):
      print("============================")
      print("Generate %dth sentences" % i)
      real, sentences, lens, summary_str = sess.run([model.real_seqs, model.fake_seqs, model.fake_lens, merge])
      writer.add_summary(summary_str, i)
      for j in range(len(real)):
        real_sens = ""
        for word_id in real[j]: 
          real_sens += model.vocab.id_to_word(word_id) + " "
        print("Real: %s" % real_sens)
        fake_sens = ""
        for k in range(lens[j]):
          fake_sens += model.vocab.id_to_word(sentences[j][k]) + " "
        print("Fake(len:%d): %s" % (lens[j], fake_sens))
    saver.save(sess, train_dir+"model.ckpt")

if __name__ == "__main__":
  tf.app.run()
