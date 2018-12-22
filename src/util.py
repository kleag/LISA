import numpy as np
import tensorflow as tf
import os


def sequence_mask_np(lengths, maxlen=None):
  if not maxlen:
    maxlen = np.max(lengths)
  return np.arange(maxlen) < np.array(lengths)[:, None]


def get_immediate_subdirectories(a_dir):
  return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def load_transitions(transition_statistics, num_classes, vocab_map):
  transition_statistics_np = np.zeros((num_classes, num_classes))
  with open(transition_statistics, 'r') as f:
    for line in f:
      tag1, tag2, prob = line.split("\t")
      transition_statistics_np[vocab_map[tag1], vocab_map[tag2]] = float(prob)
  return transition_statistics_np


def load_pretrained_embeddings(pretrained_fname):
  tf.logging.log(tf.logging.INFO, "Loading pre-trained embedding file: %s" % pretrained_fname)

  # TODO: np.loadtxt refuses to work for some reason
  # pretrained_embeddings = np.loadtxt(self.args.word_embedding_file, usecols=range(1, word_embedding_size+1))
  pretrained_embeddings = []
  with open(pretrained_fname, 'r') as f:
    for line in f:
      split_line = line.split()
      embedding = list(map(float, split_line[1:]))
      pretrained_embeddings.append(embedding)
  pretrained_embeddings = np.array(pretrained_embeddings)
  pretrained_embeddings /= np.std(pretrained_embeddings)
  return pretrained_embeddings


def get_token_take_mask(task, task_config, outputs):
  task_map = task_config[task]
  token_take_mask = None
  if "token_take_mask" in task_map:
    token_take_conf = task_map["token_take_mask"]
    token_take_mask = outputs["%s_%s" % (token_take_conf["layer"], token_take_conf["output"])]

  return token_take_mask
