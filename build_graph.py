import pickle as pkl
import random
from math import log
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from Project import project, graph_config


def read_train_test_split():
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    train_ids = []
    test_ids = []

    with open(project.train_test_split_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
                test_id = doc_name_list.index(line.strip())
                test_ids.append(test_id)
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
                train_id = doc_name_list.index(line.strip())
                train_ids.append(train_id)

    random.shuffle(train_ids)
    train_ids_str = '\n'.join(str(index) for index in train_ids)
    with open(project.shuffle_index_dir / 'train.id', 'w') as f:
        f.write(train_ids_str)

    random.shuffle(test_ids)
    test_ids_str = '\n'.join(str(index) for index in test_ids)
    with open(project.shuffle_index_dir / 'test.id', 'w') as f:
        f.write(test_ids_str)

    print("Total Train Doc No. = %s" % len(train_ids))
    print("Total Test Doc No. = %s" % len(test_ids))

    return doc_name_list, doc_train_list, doc_test_list, train_ids, test_ids


def read_content_list():
    doc_content_list = []
    with open(project.clean_corpus_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
    return doc_content_list


def get_shuffle_name_content_list(doc_name_list, doc_content_list, ids):
    shuffle_doc_name_list = []
    shuffle_doc_content_list = []
    for single_id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(single_id)])
        shuffle_doc_content_list.append(doc_content_list[int(single_id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_content_str = '\n'.join(shuffle_doc_content_list)

    with open(project.shuffle_index_dir / 'name.txt', 'w') as f:
        f.write(shuffle_doc_name_str)

    with open(project.shuffle_index_dir / 'content.txt', 'w') as f:
        f.write(shuffle_doc_content_str)

    return shuffle_doc_name_list, shuffle_doc_content_list


def get_freq_vocab(shuffle_doc_content_list):
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_content_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    vocab = list(word_set)
    word_id_map = {}
    for i in range(len(vocab)):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    with open(project.vocab_path / (project.dataset + '.txt'), 'w') as f:
        f.write(vocab_str)

    print("Vocab Size = %s" % len(vocab))

    return vocab, word_id_map, word_freq


def cal_word_doc_freq(shuffle_doc_content_list):
    word_doc_list = {}
    for i in range(len(shuffle_doc_content_list)):
        doc_words = shuffle_doc_content_list[i]
        words = doc_words.split()
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


def get_label_list(shuffle_doc_name_list):
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    with open(project.label_path / (project.dataset + '.txt'), 'w') as f:
        f.write(label_list_str)
    return label_list


def get_xy(doc_num, label_list, shuffle_doc_name_list, pre_num=0):
    word_embeddings_dim = graph_config.word_embeddings_dim
    row_x = []
    col_x = []
    data_x = []
    for i in range(doc_num):
        doc_vec = np.array([0.0 for _ in range(word_embeddings_dim)])

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j])

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        doc_num, word_embeddings_dim))

    y = []
    for i in range(doc_num):
        doc_meta = shuffle_doc_name_list[i + pre_num]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for _ in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    # [doc_num, label_size]
    y = np.array(y)
    return x, y


def get_xy_train_vocab(train_size, vocab_size, label_list, shuffle_doc_name_list):
    word_embeddings_dim = graph_config.word_embeddings_dim
    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))
    row_allx = []
    col_allx = []
    data_allx = []
    for i in range(train_size):
        doc_vec = np.array([0.0 for _ in range(word_embeddings_dim)])
        for j in range(word_embeddings_dim):
            row_allx.append(i)
            col_allx.append(j)
            data_allx.append(doc_vec[j])
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(i + train_size)
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for _ in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for _ in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)
    return allx, ally


def get_windows(shuffle_doc_content_list):
    window_size = graph_config.window_size
    windows = []
    for doc_words in tqdm(shuffle_doc_content_list, desc='get_windows'):
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
    return windows


# word_window_freq： word -> No. of appearing in the window
def get_word_window_freq(windows):
    word_window_freq = {}
    for window in tqdm(windows, 'get_word_window_freq'):
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    return word_window_freq


def get_word_pair_count(windows, word_id_map):
    word_pair_count = {}
    for window in tqdm(windows, 'get_word_pair_count'):
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    return word_pair_count


def insert_pmi(windows, word_window_freq, word_pair_count, row, col, weight, vocab, train_size):
    num_window = len(windows)

    for key in tqdm(word_pair_count, 'insert_pmi'):
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)


# "doc_id, word_id" -> No. of word_id in doc_id
def get_doc_word_freq(shuffle_doc_content_list, word_id_map):
    doc_word_freq = {}
    for doc_id in tqdm(range(len(shuffle_doc_content_list)), 'get_doc_word_freq'):
        doc_words = shuffle_doc_content_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    return doc_word_freq


def insert_tf_idf(shuffle_doc_content_list, doc_word_freq, word_doc_freq, row, col, weight,
                  train_size, vocab_info):
    vocab, word_id_map, vocab_size = vocab_info
    for i in tqdm(range(len(shuffle_doc_content_list)), 'insert_tf_idf'):
        doc_words = shuffle_doc_content_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_content_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)


def build_graph(shuffle_doc_content_list, train_size, test_size, vocab_info):
    word_doc_freq = cal_word_doc_freq(shuffle_doc_content_list)
    windows = get_windows(shuffle_doc_content_list)
    word_window_freq = get_word_window_freq(windows)
    vocab, word_id_map, vocab_size = vocab_info
    word_pair_count = get_word_pair_count(windows, word_id_map)

    row = []
    col = []
    weight = []
    insert_pmi(windows, word_window_freq, word_pair_count, row, col, weight, vocab, train_size)
    doc_word_freq = get_doc_word_freq(shuffle_doc_content_list, word_id_map)
    insert_tf_idf(shuffle_doc_content_list, doc_word_freq, word_doc_freq, row, col, weight,
                  train_size, vocab_info)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    return adj


def split_train_val(train_size, shuffle_doc_name_list):
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size
    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    with open(project.shuffle_index_dir / 'real_train.name', 'w') as f:
        f.write(real_train_doc_names_str)
    return real_train_size


def main():
    # 读入训练测试划分信息、文档内容
    doc_name_list, doc_train_list, doc_test_list, train_ids, test_ids = read_train_test_split()
    doc_content_list = read_content_list()
    ids = train_ids + test_ids
    shuffle_doc_name_list, shuffle_doc_content_list = \
        get_shuffle_name_content_list(doc_name_list, doc_content_list, ids)

    # 拆分训练集和验证集
    train_size = len(train_ids)
    real_train_size = split_train_val(train_size, shuffle_doc_name_list)
    # 构建文档特征矩阵
    label_list = get_label_list(shuffle_doc_name_list)
    x, y = get_xy(real_train_size, label_list, shuffle_doc_name_list, 0)
    test_size = len(test_ids)
    tx, ty = get_xy(test_size, label_list, shuffle_doc_name_list, train_size)

    # 构建单词特征矩阵
    vocab, word_id_map, word_freq = get_freq_vocab(shuffle_doc_content_list)
    vocab_size = len(vocab)
    allx, ally = get_xy_train_vocab(train_size, vocab_size, label_list, shuffle_doc_name_list)

    '''
    Doc word heterogeneous graph
    构建邻接矩阵
    '''
    adj = build_graph(shuffle_doc_content_list, train_size, test_size, [vocab, word_id_map, vocab_size])
    dump_to_file([x, y, tx, ty, allx, ally, adj])


def dump_to_file(graph_info):
    x, y, tx, ty, allx, ally, adj = graph_info
    save_dir = project.graph_dir
    print('x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, adj.shape')
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, adj.shape)
    # dump objects
    with open(save_dir / "x", 'wb') as f:
        pkl.dump(x, f)

    with open(save_dir / "y", 'wb') as f:
        pkl.dump(y, f)

    with open(save_dir / "tx", 'wb') as f:
        pkl.dump(tx, f)

    with open(save_dir / "ty", 'wb') as f:
        pkl.dump(ty, f)

    with open(save_dir / "allx", 'wb') as f:
        pkl.dump(allx, f)

    with open(save_dir / "ally", 'wb') as f:
        pkl.dump(ally, f)

    with open(save_dir / "adj", 'wb') as f:
        pkl.dump(adj, f)


if __name__ == '__main__':
    main()

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