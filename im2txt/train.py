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
import numpy as np

import configuration
import generative_model
import descriminative_model

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

  flags_number_of_steps = 1000000

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
    G_model = generative_model.GenerativeModel(
        model_config, mode="train", train_inception=False)
    G_model.build()
    
    D_model = descriminative_model.DescriminativeModel(model_config)
    D_model.build(G_model.output())

    G_loss = tf.reduce_mean(D_model.G_loss)
    D_loss = tf.reduce_mean(D_model.D_loss)

    tf.summary.scalar("G_loss", G_loss)
    tf.summary.scalar("D_loss", D_loss)

    G_train = tf.train.GradientDescentOptimizer(0.0002).minimize(G_loss, var_list=G_model.var_list)
    D_train = tf.train.GradientDescentOptimizer(0.0002).minimize(D_loss, var_list=D_model.var_list)
    
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
    merge = tf.summary.merge_all()

  with tf.Session(graph=g) as sess:

    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("""
======
An existing model was found in the checkpoint directory.
======
""")
    else:
      print("""
======
No existing model was found in the checkpoint directory.
Initializing a new one.
======
""")
      tf.global_variables_initializer().run()
      G_model.init_fn(sess)
      

    print("Starting the input queue runners...")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print("===Start training===")
    for step in xrange(flags_number_of_steps):

      
      _, summary_str = sess.run([G_train, merge])
      writer.add_summary(summary_str, step)

      _, summary_str = sess.run([D_train, merge])
      writer.add_summary(summary_str, step)

      _, summary_str = sess.run([D_train, merge])
      writer.add_summary(summary_str, step)
      
      if np.mod(step, 100) == 0:
        real, sentences, lens = sess.run([G_model.real_seqs, G_model.fake_seqs, G_model.fake_lens])
        for i in range(len(real)):
          real_sens = ""
          for word_id in real[i]: 
            real_sens += G_model.vocab.id_to_word(word_id) + " "
          print("Step: [%d] Real: %s" % (step, real_sens))
          fake_sens = ""
          for k in range(lens[i]):
            fake_sens += G_model.vocab.id_to_word(sentences[i][k]) + " "
          print("Fake(len:%d): %s" % (lens[i], fake_sens))
      
      if np.mod(step, 200) == 199:
        saver.save(sess, train_dir+"model.ckpt", global_step=step)

if __name__ == "__main__":
  tf.app.run()
