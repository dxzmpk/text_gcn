import pickle as pkl
import random
from math import log
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from Project import project, graph_config
import jieba
from build_graph import read_train_test_split, read_content_list


def get_freq_vocab(shuffle_doc_content_list):
    word_freq = {}
    for doc_words in shuffle_doc_content_list:
        words = jieba.lcut(doc_words)
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    read_new_string_list()
    vocab = read_vocab()
    word_id_map = {}
    for i in range(len(vocab)):
        word_id_map[vocab[i]] = i

    print("Vocab Size = %s" % len(vocab))

    return vocab, word_id_map, word_freq


def cal_word_doc_freq(doc_content_list):
    word_doc_list = {}
    for i in range(len(doc_content_list)):
        doc_words = doc_content_list[i]
        words = jieba.lcut(doc_words)
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    return word_doc_freq


# "doc_id, word_id" -> No. of word_id in doc_id
def get_doc_word_freq(new_string_list, word_id_map):
    doc_word_freq = {}
    for doc_id in tqdm(range(len(new_string_list)), 'get_doc_word_freq'):
        doc_words = new_string_list[doc_id]
        words = jieba.lcut(doc_words)
        for word in words:
            if word not in word_id_map:
                continue
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    return doc_word_freq


def get_tf_idf(new_string_list, doc_content_list, doc_word_freq, word_doc_freq, row, col, weight,
               train_size, vocab_info):
    vocab, word_id_map, vocab_size = vocab_info
    for i in tqdm(range(len(new_string_list)), 'insert_tf_idf'):
        doc_words = new_string_list[i]
        words = jieba.lcut(doc_words)
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            # To make sure the word is in the vocabulary
            if word not in word_id_map:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            if key in doc_word_freq:
                freq = doc_word_freq[key]
            else:
                freq = 0
            row.append(i)
            col.append(train_size + j)
            idf = log(1.0 * len(doc_content_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)


def build_graph(new_string_list, doc_content_list, train_size, test_size, vocab_info):
    word_doc_freq = cal_word_doc_freq(doc_content_list)
    vocab, word_id_map, vocab_size = vocab_info

    row = []
    col = []
    weight = []
    doc_word_freq = get_doc_word_freq(new_string_list, word_id_map)
    get_tf_idf(new_string_list, doc_content_list, doc_word_freq, word_doc_freq, row, col, weight,
               train_size, vocab_info)
    node_size = train_size + vocab_size + test_size
    predict_size = len(new_string_list)
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(predict_size, node_size))
    return adj


def read_new_string_list():
    doc_content_list = []
    with open(project.predict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
    return doc_content_list


def read_vocab():
    vocab = []
    with open(project.vocab_path / (project.dataset + '.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            vocab.append(line.strip())
    return vocab


def generate_predict_adj():
    # 读入训练测试划分信息、文档内容
    doc_name_list, doc_train_list, doc_test_list, train_ids, test_ids = read_train_test_split()
    new_string_list = read_new_string_list()
    doc_content_list = read_content_list()

    # 拆分训练集和验证集
    train_size = len(train_ids)
    # 构建文档特征矩阵
    test_size = len(test_ids)
    # 构建单词特征矩阵
    vocab, word_id_map, word_freq = get_freq_vocab(doc_content_list)
    vocab_size = len(vocab)

    '''
    Doc word heterogeneous graph
    构建邻接矩阵
    '''
    adj = build_graph(new_string_list, doc_content_list, train_size, test_size, [vocab, word_id_map, vocab_size])
    new_string_list_code = adj.toarray(order=None, out=None)
    print(new_string_list_code)


if __name__ == '__main__':
    generate_predict_adj()

"""
Total Train Doc No. = 2189
Total Test Doc No. = 5485
Vocab Size = 7688
get_windows: 100%|██████████| 7674/7674 [00:00<00:00, 17053.39it/s]
get_word_window_freq: 100%|██████████| 367611/367611 [00:03<00:00, 97951.28it/s]
get_word_pair_count: 100%|██████████| 367611/367611 [02:32<00:00, 2407.27it/s]
insert_pmi: 100%|██████████| 3578060/3578060 [00:08<00:00, 397825.42it/s]
get_doc_word_freq: 100%|██████████| 7674/7674 [00:00<00:00, 12855.08it/s]
insert_tf_idf: 100%|██████████| 7674/7674 [00:00<00:00, 8729.99it/s]
(2189, 300) (2189, 8) (5485, 300) (5485, 8) (9877, 300) (9877, 8) (15362, 15362)
"""