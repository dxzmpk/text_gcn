from nltk.corpus import stopwords
from Project import project, graph_config
from utils.utils import clean_str, load_word2vec
import jieba

# Read Word Vectors
def read_word_vectors():
    vocab, embedding, word_vector_map = load_word2vec(graph_config.word_vector_file)
    return vocab, embedding, word_vector_map


def read_corpus():
    doc_content_list = []
    with open(project.corpus_path, 'rb') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip().decode('latin1'))
    return doc_content_list


def cal_word_freq(doc_content_list):
    word_freq = {}
    for doc_content in doc_content_list:
        words = []
        if (project.language == 'english'):
            sentence = clean_str(doc_content)
            words = sentence.split()
        elif (project.language == 'chinese'):
            words = jieba.cut(str)
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


def remove_stop_rare(doc_content_list, word_freq):
    stop_words = set(stopwords.words(project.language))
    print(stop_words)
    clean_docs = []
    for doc_content in doc_content_list:
        sentence = clean_str(doc_content)
        words = sentence.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if project.dataset == 'mr':
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)
    return clean_docs


def corpus_statics(clean_corpus_name):
    min_len = 10000
    aver_len = 0
    max_len = 0

    with open(clean_corpus_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            temp = line.split()
            aver_len = aver_len + len(temp)
            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / len(lines)
    print('Min_len : ' + str(min_len))
    print('Max_len : ' + str(max_len))
    print('Average_len : ' + str(aver_len))


def read_clean_save_statics():
    doc_content_list = read_corpus()

    word_freq = cal_word_freq(doc_content_list)
    clean_docs = remove_stop_rare(doc_content_list, word_freq)

    clean_corpus_str = '\n'.join(clean_docs)
    with open(project.clean_corpus_path, 'w') as f:
        f.write(clean_corpus_str)
    corpus_statics(project.clean_corpus_path)


if __name__ == '__main__':
    read_clean_save_statics()

# vocab, embedding, word_vector_map = read_word_vectors()
