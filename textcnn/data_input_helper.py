import numpy as np
import re

# 导入词向量
from sklearn.preprocessing import OneHotEncoder

stopwords_file = '../data/baidu_stopwords.txt'


def load_data_and_labels(filepath, max_size=-1):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_datas = []
    stopwords_set = load_stop_words_set(stopwords_file)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        train_datas = f.readlines()
    one_hot_labels = []
    x_datas = []
    for line in train_datas:
        parts = line.split('\t', 1)
        if (len(parts[1].strip()) == 0):
            continue
        x_datas.append(pre_process(parts[1], stopwords_set))
        one_hot_labels.append(parts[0])
    enc = OneHotEncoder()
    df = enc.fit_transform(np.array(one_hot_labels).reshape(-1, 1)).toarray()
    print(' data size = ', len(train_datas))
    # Split by words
    # x_text = [clean_str(sent) for sent in x_text]
    # return [x_datas, np.array(one_hot_labels)]
    return [x_datas, df]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
            yield shuffled_data[start_index:end_index]


def get_text_idx(text, vocab, max_document_length):
    text_array = np.zeros([len(text), max_document_length], dtype=np.int32)
    for i, x in enumerate(text):
        words = x.split(" ")
        for j, w in enumerate(words):
            if w in vocab:
                text_array[i, j] = vocab[w]
            else:
                text_array[i, j] = vocab['unknown']
    return text_array


def pre_process(sentence, stop_words):
    word_list_result = [word for word in sentence.split(' ') if word not in stop_words]
    return ' '.join(word_list_result)


def load_stop_words_set(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set
