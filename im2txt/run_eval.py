
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import json

import configuration
import generative_model
from inference_utils import vocabulary

def main(unused_argv):

  flag_eval_interval_secs = 3

  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = "/media/mazm13/mscoco/test-?????-of-00008"
    model_config.vocab_file = "/media/mazm13/mscoco/word_counts.txt"
    G_model = generative_model.GenerativeModel(
        config=model_config,
        mode="eval")
    G_model.build()
    saver = tf.train.Saver()

  with tf.Session(graph=g) as sess:
    ckpt = tf.train.get_checkpoint_state("./model/train/")
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("""
=========
Loading trained model from checkpoint directory.
=========""")
    else:
      print("""
=========
Nothing was found in checkpoint directory.
=========
        """)
      return

    print("Starting the input queue runners...")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    counter = 0
    step = 0
    json_list = []
    print("===Start Evaluating===")
    for step in range(1000):
      print(step)
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))

      image_ids, real, sentences, lens = sess.run([G_model.image_ids, G_model.real_seqs, G_model.fake_seqs, G_model.fake_lens])
      
      for i in range(len(real)):
        real_sens = ""
        for word_id in real[i]: 
          real_sens += G_model.vocab.id_to_word(word_id) + " "
        #print("Image id: [%d], Real: %s" % (image_ids[i], real_sens))

        fake_sens = ""
        for k in range(1, lens[i]-1):
          word = G_model.vocab.id_to_word(sentences[i][k])
          fake_sens += word + " "
        #json_dict = {"image_id":image_ids[i], "caption":fake_sens, 'human_caption': real_sens}
        json_dict = {"image_id":image_ids[i], "caption":fake_sens}
        json_list.append(json_dict)
        #print("Fake(len:%d): %s" % (lens[i], fake_sens))
        #print("=========================")

      '''
      time_to_next_eval = start + flag_eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)
      '''
      counter += len(real)
      step += 1

    print("%d images in total were captioned")
    #with open('/home/mazm13/server/CodeIgniter-3.1.4/json/mle_caption.json', 'w') as outfile:
    with open('/home/mazm13/result/gan_caption.json', 'w') as outfile:
      json.dump(json_list, outfile)


if __name__ == "__main__":
  tf.app.run()

