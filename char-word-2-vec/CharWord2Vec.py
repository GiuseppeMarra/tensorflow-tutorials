from __future__ import print_function
from utils import rnn, data
import tensorflow as tf
import os
import numpy as np
import datetime as time
import scipy


class Model():

    def __init__(self, config, is_training, is_eval=False):

        self.x = tf.placeholder(dtype=tf.int32, shape=[None, config.word_max_len])
        chars_length = tf.reduce_sum(tf.where(self.x>0, tf.ones_like(self.x), tf.zeros_like(self.x)), axis=1)

        self.y = self.x[(config.context_size//2):-(config.context_size//2),:]

        '''Characters Embeddings'''
        embedding = tf.get_variable("embedding", [config.char_vocab_size, config.char_embed_size], dtype=tf.float32)
        chars = tf.nn.embedding_lookup(embedding, self.x)

        '''Words Embeddings'''
        self.output = rnn.morph_encoder(chars, chars_length, config.encoder_rnn_size, is_training=is_training)
        self.word_embedding = self.output

        if is_eval:
            return

        '''Contexts Embeddings'''
        indices = data.create_indices_for_context2vec(config.window_size, config.batch_num_words, skip_center=True)
        indices = tf.cast(indices, tf.int32)
        words_contexts = tf.gather(self.output, indices=indices, name="gather1")
        self.contexts = tf.reduce_sum(words_contexts, axis = 1)

        "Deconding Phase"
        labels = self.y[:,1:]
        weights = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
        logits = rnn.deconding(initial_state=self.contexts,
                               y=self.y,
                               decod_size=config.decoder_rnn_size,
                               embedding=embedding,
                               vocab_size=config.char_vocab_size,
                               batch_size=config.batch_size,
                               word_max_len=config.word_max_len,
                               dropout=config.dropout,
                               is_training=is_training)

        "Loss"
        self.predictions = tf.argmax(logits,  axis=2)
        weights = tf.cast(weights, tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            labels,
            weights,
            average_across_timesteps=True,
            average_across_batch=True
        )

        self.cost = loss
        if not is_training:
            return
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cost)



class Config():
    '''Defining some useful variables'''
    char_vocab_size = 256  # The size of the character vocabulary
    char_embed_size = 50 # The size of the character embedding
    window_size = 11  # The total size of the window [window//2 center window//2]
    if window_size % 2 == 0:   window_size += 1  # Assuring window_size is odd
    context_size = (window_size - 1)
    batch_num_words = 1024
    batch_size = batch_num_words - context_size
    cell_type = "LSTM"
    encoder_rnn_size = 500  # hidden layer of the encoder RNN
    decoder_rnn_size = 1000
    encoder_num_layers = 1
    embedding_size = None
    word_max_len = 15
    _PAD = 0
    _GO = 1
    _EOW = 2
    _UNK = 3
    shuffling = False
    learning_rate = 0.001
    beta = 0.000
    dropout = 0.5
    epochs = 25

def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)  # see what is better

    config = Config()
    with tf.variable_scope("Model", reuse=None):
        m = Model(config, is_training=True)

    with tf.variable_scope("Model", reuse=True):
        mtest = Model(config, is_training=False)

    if not os.path.exists("savings/"):
        os.makedirs("savings/")

    data_path = "datasets/mscc/data_cleaned/"
    test_file = "datasets/ptb/test/test.txt"
    test_data = data.read_words(os.path.join(test_file))
    test_data = data.build_dataset_from_char(test_data, config.word_max_len)

    save_path = "savings/CharWord2Vec2-" + data_path.split("/")[1] + "-" + str(time.datetime.now())
    os.mkdir(save_path)
    data.store_config(config, save_path)
    log_file = open(os.path.join(save_path, "log_file_train"), "w")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        list_files = [os.path.join(data_path, f)
                      for f in os.listdir(data_path)
                      if os.path.isfile(os.path.join(data_path, f))]
        print("Starting")

        def test_eval():
            testc = 0
            for idx in range(len(test_data) // config.batch_num_words):
                from_ = idx * config.batch_num_words
                to = from_ + config.batch_num_words
                test_feed_dict = {mtest.x: test_data[from_:to, :]}
                testc += sess.run(mtest.cost, feed_dict=test_feed_dict)
            log_str = "Epoch %d/%d\t File%d/%d\t \033[93m Test Cost %.3f \033[0m" % (
                e, config.epochs, k, len(list_files), testc / (len(test_data) // config.batch_num_words))
            print(log_str)
            log_str = "Epoch %d/%d\t File%d/%d\t Test Cost %.3f " % (
                e, config.epochs, k, len(list_files), testc / (len(test_data) // config.batch_num_words))
            log_file.write(log_str + "\n")

        for e in range(0, config.epochs):
            for k in range(len(list_files)):
                words_data = data.read_words(os.path.join(list_files[k]))
                words_data = data.subsampleFrequentWords(words_data)
                chars_data = data.build_dataset_from_char(words_data, config.word_max_len)
                dataset_size = len(chars_data)
                for i in range(dataset_size // config.batch_num_words):
                    from_ = i*config.batch_num_words
                    to = from_ + config.batch_num_words
                    feed_dict = {m.x: chars_data[from_:to, :]}

                    if i % 100 == 0:
                        _, c = sess.run((m.train_op, m.cost), feed_dict=feed_dict)
                        log_str = "Epoch %d/%d\t File%d/%d\t Iteration %d/%d\t Cost %.3f" %(e, config.epochs, k, len(list_files), i, dataset_size // config.batch_num_words, c)
                        print(log_str)
                        log_file.write(log_str+"\n")

                    else:
                        sess.run(m.train_op, feed_dict=feed_dict)
                    if i % 1000 == 0:
                        # test_eval()
                        saver.save(sess, os.path.join(save_path, "model.ckpt"))
                if k%10==0:
                    test_eval()
                saver.save(sess, os.path.join(save_path, "model.ckpt"))



    log_file.close()




def eval_neighbors(save_path):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:

        conf = data.restore_config(save_path)
        log_file = open(os.path.join(save_path, "log_file_neighbors"), "w")
        with tf.variable_scope("Model"):
            model = Model(conf, is_training=False, is_eval=True)

        words_data_all = data.read_words(os.path.join("datasets/ptb/full/textfull.txt"))

        words_data = list(set(words_data_all))
        k_most = data.k_frequent(words_data_all, 1000)
        valid = k_most[100:150]
        validation_data = data.indices_in(valid, words_data)

        chars_data = data.build_dataset_from_char(words_data, conf.word_max_len)

        feed_dict = {model.x: chars_data}

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(save_path, "model.ckpt"))

        embeddings = sess.run(model.word_embedding, feed_dict)

        validation_dataset = tf.constant(validation_data, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, validation_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        sim = sess.run(similarity)
        for i in range(len(valid)):
            word = valid[i]
            top_k = 10
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % word
            for k in range(top_k):
                close_word = words_data[nearest[k]]
                distance = sim[i, nearest[k]]
                log_str = "%s \n\t %s," % (log_str, close_word + "(" + str(distance) + ")")
            print(log_str)
            log_file.write(log_str+"\n")


def eval_ws353(save_path):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with  tf.Session(config=config) as sess:

        conf = data.restore_config(save_path)

        with tf.variable_scope("Model"):
            model = Model(conf,is_training=False, is_eval=True)

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(save_path, "model.ckpt"))

        # x1, x2, human_scores = data.ws353("datasets/ws353/set1.csv", conf.word_max_len)
        x1, x2, human_scores = data.rw("datasets/rw/rw.txt", conf.word_max_len)

        feed_dict_first = {model.x: x1}
        feed_dict_second = {model.x: x2}

        print(x1[0], x2[0], human_scores[0])

        xx1 = sess.run(model.word_embedding, feed_dict_first)
        xx2 = sess.run(model.word_embedding, feed_dict_second)

        xx1_norm = np.sqrt(np.sum(np.square(xx1), axis=1))
        xx2_norm = np.sqrt(np.sum(np.square(xx2), axis=1))

        scores = np.sum(np.multiply(xx1, xx2), axis=1) / np.multiply(xx1_norm, xx2_norm)
        scores = (scores - np.min(scores)) * 10 / (np.max(scores) - np.min(scores))
        mean_error = np.mean(np.abs(scores - human_scores))
        r, p = scipy.stats.spearmanr(human_scores, scores)
        rr, pp = scipy.stats.pearsonr(human_scores, scores)
        print("Spearman correlation: %.3f, PValue: %.7f" % (r, p))
        print("Pearson correlation: %.3f, PValue: %.7f" % (rr, pp))

        print("Mean error per prediction: %.3f" % mean_error)

if __name__ =='__main__':

    # save_path = "savings/CharWord2Vec2-mscc-2017-09-28 19:18:49.837291"
    # eval_neighbors(save_path)
    # eval_ws353(save_path)
    train()












