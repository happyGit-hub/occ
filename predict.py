import os
import pickle
import tensorflow as tf
from utils import create_model, get_logger,test_ner
from model import Model
from data_utils import create_input, BatchManager
from loader import load_sentences, update_tag_scheme
from loader import input_from_line,prepare_dataset
from train import FLAGS, load_config

def main(_):
    config = load_config(FLAGS.config_file)
    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    logger = get_logger(FLAGS.log_file)
    predict_sentences = load_sentences(FLAGS.predict_file, FLAGS.lower, FLAGS.zeros)
    predict_data = prepare_dataset(
        predict_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    predict_manager = BatchManager(predict_data, FLAGS.batch_size)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        re=model.evaluate(sess,predict_manager, id_to_tag)
        test_ner(re, 'data')
        # while True:
            # line = input("input sentence, please:")
            # result = model.evaluate_line(sess, input_from_line(line, FLAGS.max_seq_len, tag_to_id), id_to_tag)
            # print(result['entities'])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)