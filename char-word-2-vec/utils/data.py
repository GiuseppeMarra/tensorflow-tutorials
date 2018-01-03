import tensorflow as tf
import nltk as nl
import numpy as np
import string
from collections import Counter
import os
import pickle
import random
from math import sqrt
try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest

_PAD = 0
_GO = 1
_EOW = 2
_UNK = 3


'''
Read a file (filename) and return the textual content of the file in a vector of words
'''
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_words(filename, max_len=None):
    try:
        nl.data.find('tokenizers/punkt')
    except LookupError:
        nl.download('punkt')
    with open(filename, "r") as f:
        st = f.read()
        st = st.translate(None, string.punctuation)
        data = nl.word_tokenize(st)
        del(st)
        if max_len:
            return data[:max_len]
        return data

def subsampleFrequentWords(words):
    counts = Counter(words)
    total_length = len(words)
    new_words = []
    for w in words:
        f = counts[w]/total_length
        if f<1e-12:
            p=1
        else:
            p = (sqrt(f/0.001)+1) * 0.001 / f
        c = random.uniform(0., 1.)
        if c < p:
            new_words.append(w)
    return new_words

def read_words_from_folder(data_path):
    try:
        nl.data.find('tokenizers/punkt')
    except LookupError:
        nl.download('punkt')
    list_files = [os.path.join(data_path, f)
                  for f in os.listdir(data_path)
                  if os.path.isfile(os.path.join(data_path, f))]
    words = []
    for filename in list_files:
        with open(filename, "r") as f:
            try:
                st = f.read()
            except UnicodeDecodeError:
                print("File "+filename+" decode error: SKIPPED")
                continue
            st = st.translate(string.punctuation)
            data = nl.word_tokenize(st)
            del(st)
            words.extend(data)
    return words

def build_dataset_of_words(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def cbow_words(words, window_size):
    dataset_size=len(words)- window_size +1
    indices = np.expand_dims(range(dataset_size), 1)
    indices = indices.astype(np.int32)
    indices0 = indices
    for i in range(1, window_size):
        indices_t = np.add(indices0, i)
        indices = np.concatenate((indices, indices_t), axis=1)

    data = np.take(words, indices, axis=0)
    del (indices)
    mid = window_size // 2
    x_np = np.concatenate((data[:, :mid], data[:, (mid + 1):]), axis=1)
    y_np = data[:, mid]
    del (data)
    return x_np, y_np


def build_dataset_from_char(words, word_max_size):
    char_words = np.ndarray(shape=[len(words), word_max_size], dtype=np.int32)
    for i in range(len(words)):
        if words[i]=="<blank>":
            char_words[i][:] = _PAD
            continue
        char_words[i][0]=_GO
        for j in range(1,word_max_size):
            if j < len(words[i])+1:
                char_words[i][j] = ord(words[i][j-1])
            elif j == len(words[i])+1:
                char_words[i][j] = _EOW
            else:
                char_words[i][j] = _PAD
        if char_words[i][word_max_size-1] != _PAD:
            char_words[i][word_max_size-1] =_EOW
    return char_words

def build_char_vocab():
    vocab = {}
    vocab['_PAD']=0
    vocab['_GO'] = 1
    vocab['_EOW'] = 2
    vocab['_UNK'] = 3
    for i in range(26):
        vocab[chr(ord('A')+i)] = i+4


    for i in range(26):
        vocab[chr(ord('a')+i)] = i+30
    return vocab

def build_dataset_from_char_custom_vocab(words, word_max_size):
    char_words = np.ndarray(shape=[len(words), word_max_size], dtype=np.int32)
    vocab = build_char_vocab()
    for i in range(len(words)):
        if words[i]=="<blank>":
            char_words[i][:] = _PAD
            continue
        char_words[i][0]=_GO
        for j in range(1,word_max_size):
            if j < len(words[i])+1:
                if (words[i][j-1]) in vocab:
                    char_words[i][j] = vocab[(words[i][j-1])]
                else:
                    char_words[i][j] = vocab['_UNK']
            elif j == len(words[i])+1:
                char_words[i][j] = _EOW
            else:
                char_words[i][j] = _PAD
        if char_words[i][word_max_size-1] != _PAD:
            char_words[i][word_max_size-1] =_EOW
    return char_words


def cbow_producer_chars(chars, window_size, batch_size):
    if window_size%2==0:
        window_size += 1
    dataset_size = len(chars) - window_size +1

    word_max_size = chars.shape[1]
    indices = np.expand_dims(range(dataset_size), 1)
    indices0=indices
    for i in range(1,window_size):
        indices_t = np.add(indices0, i)
        indices = np.concatenate((indices, indices_t), axis=1)
    data = np.take(chars, indices, axis=0)
    mid = window_size//2
    x_np = np.concatenate((data[:,:mid,:],data[:,(mid+1):, :]), axis = 1)
    y_np= data[:,mid,:]

    epoch_size = dataset_size//batch_size
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(x_np.astype(np.int32), [i*batch_size, 0, 0],
                         [(i+1)*batch_size, window_size-1,word_max_size ])
    x.set_shape([batch_size, window_size-1, word_max_size])
    y = tf.strided_slice(y_np.astype(np.int32), [i*batch_size, 0],
                         [(i+1)*batch_size, word_max_size])
    y.set_shape([batch_size, word_max_size])
    return x, y

def cbow_feed_chars(chars, window_size):
    if window_size%2==0:
        window_size += 1
    dataset_size = len(chars) - window_size +1

    word_max_size = chars.shape[1]
    indices = np.expand_dims(range(dataset_size), 1)
    indices = indices.astype(np.int32)
    indices0=indices
    for i in range(1,window_size):
        indices_t = np.add(indices0, i)
        indices = np.concatenate((indices, indices_t), axis=1)

    data = np.take(chars, indices, axis=0)
    del(indices)
    mid = window_size//2
    x_np = np.concatenate((data[:,:mid,:],data[:,(mid+1):, :]), axis = 1)
    y_np= data[:,mid,:]
    del(data)

    return x_np, y_np


def cbow_data_from_file(filename, word_max_len, window_size):
    '''Reading the input file as a list of words'''
    #words_data = data.read_words(os.path.join("datasets/shakespeare/t8.shakespeare_short.txt"))
    words_data = read_words(os.path.join(filename))


    '''Translating the list of words in a matrices of characters.
    Each characters is substituted by an integer, and words are preponed and postponed with the GO and EOW sybols
    respectively. Words are padded/truncated to a maximum length'''
    #chars_data = build_dataset_from_char_custom_vocab(words_data, word_max_len)
    chars_data = build_dataset_from_char(words_data, word_max_len)

    dataset_size = len(chars_data) - window_size +1

    xd, yd = cbow_feed_chars(chars_data, window_size)
    del chars_data
    del words_data
    return xd, yd, dataset_size

def cbow_data_from_file_line_by_line(filename, word_max_len, window_size):
    '''Reading the input file as a list of words'''
    dataset_size = 0
    xd = None
    yd = None
    with open(os.path.join(filename)) as f:
        for line in open(os.path.join(filename)):
            line_words_data = line.split()
            chars_data = build_dataset_from_char(line_words_data, word_max_len)
            dataset_size += (len(chars_data) - window_size + 1)
            xdt, ydt = cbow_feed_chars(chars_data, window_size)
            xd = xdt if type(xd) is not np.ndarray else np.concatenate((xd,xdt), 0)
            yd = ydt if type(yd) is not np.ndarray else np.concatenate((yd, ydt), 0)

    return xd, yd, dataset_size

def cbow_data_from_binary_file(filename, word_max_len, window_size):
    '''Reading the input file as a list of words'''
    #words_data = data.read_words(os.path.join("datasets/shakespeare/t8.shakespeare_short.txt"))
    words_data = read_words(os.path.join(filename))


    '''Translating the list of words in a matrices of characters.
    Each characters is substituted by an integer, and words are preponed and postponed with the GO and EOW sybols
    respectively. Words are padded/truncated to a maximum length'''
    chars_data = build_dataset_from_char(words_data, word_max_len)

    dataset_size = len(chars_data) - window_size +1

    xd, yd = cbow_feed_chars(chars_data, window_size)
    del chars_data
    del words_data
    return xd, yd, dataset_size

def k_frequent(words_data, k):
    counter = Counter(words_data)
    most = counter.most_common(k)
    res = [most[i][0] for i in range(len(most))]
    return res


def dummy(fname, window_size, word_max_size, batch_size):


    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    finx =np.ones(shape=(1,2,10))
    finy =np.ones(shape=(1,10))
    for x in content:

        x = x.split()
        dataset_size=len(x) - window_size +1
        chars = build_dataset_from_char(x,word_max_size )

        indices = np.expand_dims(range(dataset_size), 1)
        indices0=indices
        for i in range(1,window_size):
            indices_t = np.add(indices0, i)
            indices = np.concatenate((indices, indices_t), axis=1)

        data = np.take(chars, indices, axis=0)
        mid = window_size // 2
        x_np = np.concatenate((data[:, :mid, :], data[:, (mid + 1):, :]), axis=1)
        y_np = data[:, mid, :]

        finx = np.concatenate((finx, x_np), axis=0)
        finy = np.concatenate((finy, y_np), axis=0)

    finx = finx[1:,:,:]
    finy = finy[1:,:]
    print(finx)
    print(finy)
    dataset_size = finx.shape[0]
    epoch_size = dataset_size // batch_size -1
    i = tf.train.range_input_producer(epoch_size-1, shuffle=False).dequeue()
    x = tf.strided_slice(finx.astype(np.int32), [i * batch_size, 0, 0],
                         [(i + 1) * batch_size, window_size - 1, word_max_size])
    x.set_shape([batch_size, window_size - 1, word_max_size])
    y = tf.strided_slice(finy.astype(np.int32), [i * batch_size, 0],
                         [(i + 1) * batch_size, word_max_size])
    y.set_shape([batch_size, word_max_size])
    return x, y, dataset_size

def indices_in(some, all):
    all_dict = {k: v for v, k in enumerate(all)}
    return [all_dict[some[i]] for i in range(len(some))]

def multiple_indices_in(some, all):
    res = []
    for v in some:
        for j, w in enumerate(all):
            if v==w:
                res.append(j)
    return res




# def toWord(chars):
#     inverse = {v: k for k, v in build_char_vocab().items()}
#     str =''
#     for c in chars:
#         if(c<2):
#             continue
#         elif(c==2):
#             break
#         str = str+inverse[c]
#     return str

def toWord(chars):
    str =''
    for c in chars:
        if(c<2):
            continue
        elif(c==2):
            break
        str = str+chr(c)
    return str


def serialize_dataset(data_path, word_max_len,window_size):
    list_files = [os.path.join(data_path, f)
                  for f in os.listdir(data_path)
                  if os.path.isfile(os.path.join(data_path, f))]

    for file in list_files:
        writer = tf.python_io.TFRecordWriter(file+".raw")
        xd, yd, dataset_size = cbow_data_from_file(file,word_max_len,window_size)
        for i in range(xd.shape[0]):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'y': _bytes_feature(yd[i].astype("int64").tostring()),
                        'x': _bytes_feature(xd[i].astype("int64").tostring()),
                    }))
            writer.write(example.SerializeToString())
        writer.close()

def read_and_decode_batch_example(filename_list,word_max_len,window_size, batch_size):


    # filename_list = [os.path.join(data_path, f)
    #               for f in os.listdir(data_path)
    #               if os.path.isfile(os.path.join(data_path, f))]

    filename_queue = tf.train.string_input_producer(filename_list,
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'y': tf.FixedLenFeature([], tf.string),
            'x': tf.FixedLenFeature([], tf.string)
        })
    x = tf.decode_raw(features['x'],tf.int64)
    x.set_shape([(window_size-1)*word_max_len])
    x = tf.reshape(x, [window_size-1,word_max_len] )
    y = tf.decode_raw(features['y'], tf.int64)
    y.set_shape([word_max_len])

    xb, yb = tf.train.shuffle_batch(
        [x, y],
        batch_size=batch_size,
        num_threads=10,
        capacity=10000,
        min_after_dequeue=1000)
    return xb, yb


def storeNumpyCBOW(filename,word_max_len, window_size):
    words_data = read_words(os.path.join(filename))

    chars_data = build_dataset_from_char(words_data, word_max_len)

    dataset_size = len(chars_data) - window_size +1

    xd, yd = cbow_feed_chars(chars_data, window_size)

    np.save(filename+"-xd.npy", xd)
    np.save(filename+"-yd.npy", yd)

def restoreNumpyCBOW(filename):
    xd = np.load(filename+"-xd.npy")
    yd = np.load(filename+"-yd.npy")
    return xd,yd


def testQueueFuntion(data_path, word_max_len, window_size, batch_size):
    filename_list = [os.path.join(data_path, f)
                  for f in os.listdir(data_path)
                  if os.path.isfile(os.path.join(data_path, f))]
    filename_queue = tf.train.string_input_producer(filename_list)


class Example:

    def __init__(self, id, question, answers, position):
        self.id=id
        self.question=question
        self.answers=answers
        self.position = position
        self.context = None

def readMSCCTestData(filename):
    examples = []

    f = open(filename, "r")
    for i,line in enumerate(f):
        if i==0:
            continue
        line = line.replace("_____","XXXX")
        splitted = line.split(",")
        if "\"" in line:
            question = line.split("\"")[1]
        else:
            question = splitted[1]
        id = splitted[0]
        question = question.translate(None, string.punctuation)
        question = question.split()
        position= None
        for i, w in enumerate(question):
            if w=="XXXX":
                question.remove(w)
                position = i
                break
        answers=[]
        answers.append(splitted[-5])
        answers.append(splitted[-4])
        answers.append(splitted[-3])
        answers.append(splitted[-2])
        answers.append(splitted[-1].translate(None, string.whitespace))
        ex = Example(id, question,answers,position)
        examples.append(ex)

    return examples

def transformMSCCIntoChars(examples, context_size, word_max_len):
    '''Extracting only windw_size word from sentence'''
    contexts = []
    answers = []
    for t,e in enumerate(examples):
        l = len(e.question)
        answers.extend(e.answers)
        if l<=context_size:
            for i in range(context_size-l):
                e.question.append("<blank>")
                contexts.append(e.question)
        else:
            context = []
            forward = e.position
            backward = e.position -1
            while len(context)<context_size:
                if forward<len(e.question):
                    context.append( e.question[forward])
                    forward+=1
                if backward >= 0:
                    context.insert(0,e.question[backward])
                    backward-=1
            contexts.append(context)



    '''Translating words into char indexes and epeating contexts 5 times, one for each answer'''
    npcontext = build_dataset_from_char_custom_vocab(contexts[0], word_max_len)
    npcontexts = np.expand_dims(npcontext, axis = 0)
    npcontexts = np.tile(npcontexts, (5,1,1))
    for i in range(1,len(contexts)):
        npcontext = build_dataset_from_char_custom_vocab(contexts[i], word_max_len)
        npcontext = np.expand_dims(npcontext, axis = 0)
        npcontext = np.tile(npcontext, (5,1,1))
        npcontexts = np.concatenate((npcontexts,npcontext), axis=0 )


    '''Translating answers into char indexes'''
    npanswers = build_dataset_from_char_custom_vocab(answers, word_max_len)
    return npcontexts, npanswers

def extractMSCCSingleQuestion(npcontexts, npanswers, i):
    return npcontexts[i*5:(i+1)*5,:,:], npanswers[i*5:(i+1)*5,:]

def readMSCCTestRightAnswers(filename):
    file = open(filename)
    results = []
    for i, line in enumerate(file):
        if i!=0:
            res = line.split(",")
            id = res[0]
            answer = res[1][0]
            results.append((id,answer))
    return results

def getMSCCScore(results, answers):
    score=0.
    for i in range(len(results)):
        if results[i][1] == answers[i][1]:
            score+=1
    return score/len(results)


def getPerWordLength(yd, batch_size, word_max_len):
    answers_length = []
    for i in range(batch_size):
        len = 0
        for k in range(word_max_len):
            if yd[i][k]==_EOW:
                break
            len+=1
        answers_length.append(len)
    return answers_length


def ws353(filename, word_max_size):
    with open(filename, 'r') as f:
        x1 = []
        x2 = []
        scores = []
        for i, line in enumerate(f):
            if i!= 0:
                splits = line.split(',')
                x1.append(splits[0])
                x2.append(splits[1])
                scores.append(float(splits[2]))
        x1 = build_dataset_from_char(x1, word_max_size)
        x2 = build_dataset_from_char(x2, word_max_size)

        return x1,x2,scores

def ws353_words(filename):
    with open(filename, 'r') as f:
        x1 = []
        x2 = []
        scores = []
        for i, line in enumerate(f):
            if i!= 0:
                splits = line.split(',')
                x1.append(splits[0])
                x2.append(splits[1])
                scores.append(float(splits[2]))

        return x1,x2,scores


def rw(filename, word_max_size):
    with open(filename, 'r') as f:
        x1 = []
        x2 = []
        scores = []
        for i, line in enumerate(f):
            if i!= 0:
                splits = line.split('\t')
                x1.append(splits[0])
                x2.append(splits[1])
                scores.append(float(splits[2]))
        x1 = build_dataset_from_char(x1, word_max_size)
        x2 = build_dataset_from_char(x2, word_max_size)

        return x1,x2,scores

def rw_words(filename):
    with open(filename, 'r') as f:
        x1 = []
        x2 = []
        scores = []
        for i, line in enumerate(f):
            if i!= 0:
                splits = line.split('\t')
                x1.append(splits[0])
                x2.append(splits[1])
                scores.append(float(splits[2]))

        return x1,x2,scores

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def generateWord2VecBatch(lines):
    words = []
    embeddings = []
    for line in lines:
        if line is not None:
            splits = line.split()
            words.append(splits[0])
            embeddings.append([float(splits[k]) for k in range(1, len(splits))])
    return words, embeddings


def createDictionaryFromMultipleFiles(data_path, vocabulary_size):
    count = [['UNK', -1]]
    counter = Counter()
    for filename in os.listdir(os.path.join(data_path)):
            words = read_words(os.path.join(data_path,filename))
            counter.update(words)
    count.extend(counter.most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary


def createWordDataIndexFromDictionary(words, dictionary):
    if isinstance(words, str):
        words = read_words(words)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
        data.append(index)
    return data


def store_config(config, save_path):
    with open(os.path.join(save_path, "config.txt"), 'w+') as config_file:
        pickle.dump(config, config_file, pickle.HIGHEST_PROTOCOL)

def restore_config(save_path):
    return pickle.load(open(os.path.join(save_path,"config.txt"),'r'))

def create_indices_for_context2vec(window_size, batch_num_words, skip_center=False):
    ws = window_size
    l = range(ws)
    if skip_center: l.remove(l[ws//2])
    l = np.array(l)
    res = l = np.array([l])
    for i in range(1, batch_num_words-ws+1):
            res = np.concatenate((res, np.add(l, i)), 0)
    return res




