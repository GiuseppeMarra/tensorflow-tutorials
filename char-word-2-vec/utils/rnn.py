import tensorflow as tf
import utils

def cell(size, type, dropout=None, proj=None, is_training = True):
    cell = None
    if type == "LSTM":
        cell= tf.contrib.rnn.BasicLSTMCell(size)
    elif type == "GRU":
        cell= tf.contrib.rnn.GRUCell(size)
    if dropout is not None and is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=1.0, state_keep_prob=1.0)
    if proj:
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, proj)
    return cell

def morph_encoder(chars, chars_length, size, cell_type="LSTM", dropout=None, is_training=True):
    '''Here we take a batch of words and compute their morphological embeddings, i.e. a hidden representation
    of a RNN over their characters'''
    with tf.variable_scope("MorphologicEncoder"):
        with tf.variable_scope("fw"):
            char_rnn_cell_fw = cell(size, cell_type, dropout, is_training=is_training)
        with tf.variable_scope("bw"):
            char_rnn_cell_bw = cell(size, cell_type, dropout, is_training=is_training)
        _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=char_rnn_cell_fw,
                                                               cell_bw=char_rnn_cell_bw,
                                                               inputs=chars,
                                                               sequence_length=chars_length,
                                                               dtype=tf.float32)

    return  tf.concat((fw_state.h, bw_state.h), axis=1)


def deconding(initial_state,decod_size, vocab_size, embedding, y, word_max_len, batch_size, cell_type="LSTM", dropout=1.0, is_training=True):
    '''Decoding Phase'''
    decoder_input = initial_state
    initial_decoder_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros_like(decoder_input),
                                                          h=decoder_input)
    decoder_cell_fw = cell(decod_size, cell_type, dropout=dropout, is_training = is_training)

    decoder_inputs = tf.nn.embedding_lookup(embedding, y[:, :(word_max_len - 1)])
    decoder_inputs = tf.unstack(decoder_inputs, axis=1)
    final_outputs, _ = tf.nn.static_rnn(cell=decoder_cell_fw,
                                        dtype=tf.float32,
                                        inputs=decoder_inputs,
                                        initial_state=initial_decoder_state)

    W = tf.get_variable("softmax_w", [decod_size, vocab_size], dtype=tf.float32)
    b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

    output = tf.reshape(tf.stack(axis=1, values=final_outputs), [-1, decod_size])

    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, word_max_len - 1, vocab_size])

    return logits

def dynamic_deconding(initial_state, decod_size, vocab_size, embedding, y, word_max_len, batch_size, sequence_length=None, cell_type="LSTM"):
    '''Decoding Phase'''
    decoder_input = initial_state
    initial_decoder_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros_like(decoder_input),
                                                          h=decoder_input)
    decoder_cell_fw = cell(decod_size, cell_type, dropout=0.5)

    decoder_inputs = tf.nn.embedding_lookup(embedding, y[:, :(word_max_len - 1)])
    final_outputs, _ = tf.nn.dynamic_rnn(
        cell=decoder_cell_fw,
        inputs=decoder_inputs,
        sequence_length=sequence_length,
        initial_state=initial_decoder_state)


    W = tf.get_variable("softmax_w", [decod_size, vocab_size], dtype=tf.float32)
    b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

    output = tf.reshape(final_outputs, [-1, decod_size])

    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, word_max_len - 1, vocab_size])

    return logits

# def encode_words(chars, position, encoder_cell_fw, encoder_cell_bw, encoder_rnn_size, embedding_size, reuse=True):
#     with tf.variable_scope("EmbeddingEncoder", reuse=reuse):
#         if embedding_size is not None:
#             We = tf.get_variable("embedding_w", [2 * encoder_rnn_size, embedding_size], dtype=tf.float32)
#             be = tf.get_variable("embedding_b", [embedding_size], dtype=tf.float32)
#         with tf.variable_scope("BRNN_Encoder", reuse=reuse):
#             input = chars[:, position, :, :]
#             input = tf.squeeze(input)
#             _, word_k = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
#                                                         cell_bw=encoder_cell_bw,
#                                                         dtype=tf.float32,
#                                                         inputs=input)
#             word_embedding = tf.concat((word_k[0][1], word_k[1][1]), axis=1)
#             if embedding_size is not None:
#                 word_embedding = tf.matmul(word_embedding, We) + be
#             return word_embedding

def encode_words(chars, position, encoder_cell_fw, encoder_cell_bw, encoder_rnn_size, embedding_size, reuse=True):
    with tf.variable_scope("EmbeddingEncoder", reuse=reuse):
        if embedding_size is not None:
            We = tf.get_variable("embedding_w", [2 * encoder_rnn_size, embedding_size], dtype=tf.float32)
            be = tf.get_variable("embedding_b", [embedding_size], dtype=tf.float32)
        with tf.variable_scope("BRNN_Encoder", reuse=reuse):
            input = chars[:, position, :, :]
            input = tf.unstack(input, axis=1)
            _, state_fw, state_bw = tf.nn.static_bidirectional_rnn(cell_fw=encoder_cell_fw,
                                                        cell_bw=encoder_cell_bw,
                                                        dtype=tf.float32,
                                                        inputs=input)
            word_embedding = tf.concat((state_fw[1], state_bw[1]), axis=1)
            return word_embedding

def encode_words_with_convolutions(chars, position, char_embed_size, filters, filter_sizes, word_max_len, reuse=True):
    with tf.variable_scope("Convolutional_Encoder", reuse=reuse):
        input = chars[:, position, :, :]
        input = tf.expand_dims(input, axis=1)
        embedding = None
        for i,filter_width in enumerate(filters):
            kernels = tf.get_variable("conv_filters"+str(filter_width), [1, filter_width, char_embed_size, filter_sizes[i]], dtype=tf.float32)
            conv = tf.nn.conv2d(input,kernels,[1,1,1,1],"VALID")
            biases = tf.get_variable("biases"+str(filter_width), [filter_sizes[i]], dtype=tf.float32)
            conv = tf.nn.relu(conv + biases)
            #conv = tf.expand_dims(conv, axis=2)
            pool = tf.nn.max_pool(conv, ksize=[1, 1,word_max_len-filter_width+1,  1], strides=[1, 1, 1, 1],padding='VALID')
            if embedding is None:
                embedding = tf.squeeze(pool)
            else:
                embedding = tf.concat((embedding, tf.squeeze(pool)), axis=1)
        return embedding


def encode_words_multilayer(chars, position, encoder_cell_fw, encoder_cell_bw, encoder_rnn_size, embedding_size, encoder_num_layers=1,reuse=True):
    with tf.variable_scope("EmbeddingEncoder", reuse=reuse):
        if embedding_size is not None:
            We = tf.get_variable("embedding_w", [2 * encoder_rnn_size, embedding_size], dtype=tf.float32)
            be = tf.get_variable("embedding_b", [embedding_size], dtype=tf.float32)
        with tf.variable_scope("BRNN_Encoder", reuse=reuse):
            input = chars[:, position, :, :]
            input = tf.squeeze(input)
            _, word_k = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                        cell_bw=encoder_cell_bw,
                                                        dtype=tf.float32,
                                                        inputs=input)
            word_embedding = tf.concat((word_k[0][-1][1], word_k[1][-1][1]), axis=1)
            if embedding_size is not None:
                word_embedding = tf.matmul(word_embedding, We) + be
            return word_embedding

'''Here we are using the legacy library tf.contrib.legacy_seq2seq'''
def decode_into_target_word(decoder_cell_fw, embedding, initial_state, y):

    decoder_inputs = tf.nn.embedding_lookup(embedding, y)
    decoder_inputs = tf.unstack(decoder_inputs,axis=1)
    return tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs,
                                              initial_state,
                                              decoder_cell_fw,
                                              loop_function=None)

def decode_into_target_word_with_convolutions(embedding, char_embed_size,filters, filter_sizes, word_max_len, reuse=True):
    embedding = tf.expand_dims(embedding, axis=1)
    with tf.variable_scope("Convolutional_AutoEncoder", reuse=reuse):
        res = None
        for i, filter_width in enumerate(filters):
            fr = sum(filter_sizes[:i])
            to = fr + filter_sizes[i] -1
            kernels = tf.get_variable("conv_filters" + str(filter_width),
                                      [1, filter_width, char_embed_size, filter_sizes[i]], dtype=tf.float32)
            conv = tf.nn.conv2d_transpose(
                embedding[:,:,fr:to],
                kernels,
                tf.stack([tf.shape(embedding)[0], 1, word_max_len, filter_sizes[i]]),
                [1,1,1,1]
            )
            print(conv)
            biases = tf.get_variable("biases_decoding" + str(filter_width), [char_embed_size], dtype=tf.float32)
            conv = tf.nn.relu(conv + biases)
            if res is None:
                res = conv
            else:
                res = tf.concat((res, conv), axis=2)

        return res




