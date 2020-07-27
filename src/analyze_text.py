import tensorflow as tf
import numpy as np
import argparse
import train_utils
from vocab import Vocab
import sys
from tensorflow.contrib import predictor
import evaluation_fns_np as eval_fns
import constants
import os
import util
import spacy
import tempfile
import time

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--test_files',
                        help='Comma-separated list of test data files')
arg_parser.add_argument('--dev_files',
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory containing saved model')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
# todo: are these necessary?
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')

arg_parser.add_argument('--ensemble', dest='ensemble', action='store_true',
                        help='Whether to ensemble models in save dir.')

arg_parser.set_defaults(debug=False, ensemble=False)

args, leftovers = arg_parser.parse_known_args()

util.init_logging(tf.logging.INFO)
tf.logging.log(tf.logging.INFO, f"Logging initialized")

if not os.path.isdir(args.save_dir):
  util.fatal_error("save_dir not found: %s" % args.save_dir)

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)
tf.logging.log(tf.logging.INFO, f"Configurations loaded")

# attention_config = {}
# if args.attention_configs and args.attention_configs != '':
#   attention_config = train_utils.load_json_configs(args.attention_configs)
layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config)

dev_filenames = args.dev_files.split(',')
test_filenames = args.test_files.split(',') if args.test_files else []
text_filenames = leftovers if leftovers else []

vocab = Vocab(data_config, args.save_dir)
vocab.update(test_filenames)
tf.logging.log(tf.logging.INFO, f"Vocabulary initialized")

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]

nlp = spacy.load('en')
tf.logging.log(tf.logging.INFO, f"Spacy initialized")

# Generate mappings from feature/label names to indices in the model_fn inputs
# feature_idx_map = {}
# label_idx_map = {}
feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)
# # todo put this in a function
# for i, f in enumerate([d for d in data_config.keys() if
#                        ('feature' in data_config[d] and data_config[d]['feature']) or
#                        ('label' in data_config[d] and data_config[d]['label'])]):
#   if 'feature' in data_config[f] and data_config[f]['feature']:
#     feature_idx_map[f] = i
#   if 'label' in data_config[f] and data_config[f]['label']:
#     if 'type' in data_config[f] and data_config[f]['type'] == 'range':
#       idx = data_config[f]['conll_idx']
#       j = i + idx[1] if idx[1] != -1 else -1
#       label_idx_map[f] = (i, j)
#     else:
#       label_idx_map[f] = (i, i+1)

# create transition parameters if training or decoding with crf/viterbi
# need to load these here for ensembling (they're also loaded by the model)
transition_params = util.load_transition_params(layer_task_config, vocab)

if args.ensemble:
    predict_fns = [
        predictor.from_saved_model(f"{args.save_dir}/{subdir}")
        for subdir in util.get_immediate_subdirectories(args.save_dir)]
else:
    predict_fns = [predictor.from_saved_model(args.save_dir)]


#def dev_input_fn():
  #return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False,
                                  #embedding_files=embedding_files)

#def convert_to_reference_format(text):
  #doc = nlp(text)
  #lines = []
  #for sentence_id, sentence in enumerate(doc.sents):
    #tokens = {}
    #for token_id, token in enumerate(sentence):
      #tokens[token] = token_id
    #for token_id, token in enumerate(sentence):
      ##0:domain  1:sent_id 2:id  3:word+word_type  4:gold_pos    5:auto_pos    6:parse_head  7:parse_label _

      #lines.append(f'conll05\t{sentence_id}\t{token_id}\t{token.text}\t{token.pos_}\t{token.tag_}\t{tokens[token.head]}\t{token.dep_}\t_\t-\t-\t-\t-\tO')
    #lines.append('')
  #return '\n'.join(lines)

def eval_fn(input_op, sess):
    eval_accumulators = eval_fns.get_accumulators(task_config)
    eval_results = {}
    i = 0
    while True:
        i += 1
        try:
            # input_np = sess.run(dev_input_fn())
            input_np = sess.run(input_op)
            predictor_input = {'input': input_np}
            predictions = [
                predict_fn(predictor_input) for predict_fn in predict_fns]

            shape = input_np.shape
            batch_size = shape[0]
            batch_seq_len = shape[1]

            feats = {
                f: input_np[:, :, idx] for f, idx in feature_idx_map.items()}
            tokens_to_keep = np.where(
                feats['word'] == constants.PAD_VALUE, 0, 1)

            combined_predictions = predictions[0]

            # todo: implement ensembling
            combined_scores = {
                k: v for k, v in combined_predictions.items() if k.endswith("_scores")}
            combined_probabilities = {
                k: v for k, v in combined_predictions.items() if k.endswith("_probabilities")}

            combined_predictions.update(
                {
                    k.replace('scores', 'predictions'):
                        np.argmax(v, axis=-1)
                        for k, v in combined_scores.items()})
            combined_predictions.update(
                {
                    k.replace('probabilities', 'predictions'):
                        np.argmax(v, axis=-1)
                        for k, v in combined_probabilities.items()})

            for task, tran_params in transition_params.items():
                task_predictions = np.empty_like(
                    combined_predictions['%s_predictions' % task])
                token_take_mask = util.get_token_take_mask(
                    task, task_config, combined_predictions)
                if token_take_mask is not None:
                    toks_to_keep_tiled = np.reshape(
                        np.tile(tokens_to_keep, [1, batch_seq_len]),
                        [batch_size, batch_seq_len, batch_seq_len])
                    toks_to_keep_task = toks_to_keep_tiled[
                        np.where(token_take_mask == 1)]
                else:
                    toks_to_keep_task = tokens_to_keep
                sent_lens_task = np.sum(toks_to_keep_task, axis=-1)
                if 'srl' in transition_params:
                    for idx, (sent, sent_len) in enumerate(
                        zip(combined_scores[f'{task}_scores'],
                            sent_lens_task)):
                        (viterbi_sequence,
                         score) = tf.contrib.crf.viterbi_decode(
                            sent[:sent_len], tran_params)
                        task_predictions[idx, :sent_len] = viterbi_sequence
                combined_predictions[f'{task}_predictions'] = task_predictions

            labels = {}
            for l, idx in label_idx_map.items():
                these_labels = (input_np[:, :, idx[0]:idx[1]]
                                if idx[1] != -1
                                else input_np[:, :, idx[0]:])
                these_labels_masked = np.multiply(
                    these_labels, np.expand_dims(tokens_to_keep, -1))
                # check if we need to mask another dimension
                if idx[1] == -1:
                    this_mask = np.where(
                        these_labels_masked == constants.PAD_VALUE, 0, 1)
                    these_labels_masked = np.multiply(these_labels_masked,
                                                      this_mask)
                else:
                    these_labels_masked = np.squeeze(these_labels_masked, -1)
                labels[l] = these_labels_masked

            # for i in layer_task_config:
            for task, task_map in task_config.items():
                for eval_name, eval_map in task_map['eval_fns'].items():
                    eval_fn_params = eval_fns.get_params(task, eval_map,
                                                         combined_predictions,
                                                         feats, labels,
                                                         vocab.reverse_maps,
                                                         tokens_to_keep)
                    eval_fn_params['accumulator'] = eval_accumulators[
                        eval_name]
                    eval_result = eval_fns.dispatch(
                        eval_map['name'])(**eval_fn_params)


                    # write predicted labels
                    # called by conll_srl_eval called called by conll_srl_eval_np dispatched to by eval_fns.dispatch

                    #def conll_srl_eval(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                            #pred_srl_eval_file, gold_srl_eval_file, pos_predictions=None, pos_targets=None):
                    # TODO create temp file temp_out, then call write_srl_eval then write file content to stdout
                    # TODO Generate/retriev words, predicate_predictions, sent_lens and srl_predictions
                    #write_srl_eval(temp_out, words, predicate_predictions, sent_lens, srl_predictions)
                    # Or use write_srl_debug
                    # write_srl_debug(filename, words, predicates, sent_lens, role_labels, pos_predictions, pos_targets)
                    #str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
                    #str_predictions = [list(map(reverse_maps['parse_label'].get, s)) for s in predictions]
                    #str_targets = [list(map(reverse_maps['parse_label'].get, s)) for s in targets]
                    #str_pos_targets = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_targets]

                    #write_srl_debug(temp_out, words, predicates, sent_lens, role_labels, pos_predictions, pos_targets)

                    #eval_results[eval_name] = eval_result
        except tf.errors.OutOfRangeError:
            break

    tf.logging.log(tf.logging.INFO, f"analyze_text.eval_fn result: {eval_results}")


with tf.Session() as sess:

    #dev_input_op = dev_input_fn()

    #test_input_ops = {}
    #for test_file in test_filenames:
        #def test_input_fn():
            #return train_utils.get_input_fn(vocab, data_config, [test_file], hparams.batch_size, num_epochs=1, shuffle=False,
                                      #embedding_files=embedding_files)
        #test_input_ops[test_file] = test_input_fn()

    # Must convert input text file in CoNLL05 files by tokenizing and splitting
    # in sentences. Use spacy ?
    tokenized_files = []
    tokenized_filenames = []
    for text_file in text_filenames:
        temp = tempfile.NamedTemporaryFile(mode='w+t')
        with open(text_file, 'r') as f:
            text = f.read()

            doc = nlp(text)
            for sentence_id, sentence in enumerate(doc.sents):
                tokens = {}
                for token_id, token in enumerate(sentence):
                    tokens[token] = token_id
                for token_id, token in enumerate(sentence):
                    if len(token.text) > 0 and token.text != '\n':
                        # 0:domain  1:sent_id 2:id  3:word+word_type
                        # 4:gold_pos 5:auto_pos    6:parse_head  7:parse_label

                        line = (f'conll05\t{sentence_id}\t{token_id}'
                                f'\t{token.text}\t{token.pos_}\t{token.tag_}'
                                f'\t{tokens[token.head]}\t{token.dep_}'
                                f'\t-\t-\t-\t-\t-\t-\t-\t-\n')
                        #line = f'conll05\t{sentence_id}\t{token_id}
                        #\t{token.text}\t_\t_\t_\t_\n'
                        #line = f'{token_id}\t{token.text}\t_\t_\t_\t_\t_\t_\n'
                        tf.logging.log(tf.logging.INFO,
                                       f"Writing to temp file {temp.name}: "
                                       f"{line}")
                        temp.write(line)
            temp.flush()

        tokenized_files.append(temp)
        tokenized_filenames.append(temp.name)

    text_input_ops = {}
    for tokenized_filename in tokenized_filenames:
        def text_input_fn():
            return train_utils.get_input_fn(vocab,
                                            data_config,
                                            [tokenized_filename],
                                            hparams.batch_size, num_epochs=1,
                                            shuffle=False,
                                            embedding_files=embedding_files)
        text_input_ops[tokenized_filename] = text_input_fn()

    sess.run(tf.tables_initializer())

    #for test_file, test_input_op in test_input_ops.items():
        #tf.logging.log(tf.logging.INFO, "Evaluating on test file: %s" % str(test_file))
        #eval_fn(test_input_op, sess)

    for tokenized_filename, text_input_op in text_input_ops.items():
        tf.logging.log(tf.logging.INFO,
                       f"Analyzing text file: {tokenized_filename}")
        eval_fn(text_input_op, sess)
    #tf.logging.log(tf.logging.INFO, f"Pausing 20 seconds")
    #time.sleep(20)
    for tokenized_file in tokenized_files:
        tokenized_file.close()
