from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    dataset = 'R8'
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'data'

    corpus_path = data_dir / 'corpus' / (dataset + '.txt')
    train_test_split_path = data_dir / 'train_val_split' / (dataset + '.txt')
    clean_corpus_path = data_dir / 'corpus/clean' / (dataset + '.clean.txt')
    shuffle_index_dir = data_dir / 'shuffle/'
    vocab_path = data_dir / 'vocab/'
    label_path = data_dir / 'label/'
    graph_dir = data_dir / 'graph' / (dataset + '/')

    # checkpoint_dir = base_dir / 'checkpoint'
    # precess_dir = data_dir / 'precess'
    # train_dir = data_dir / 'train/ohsumed_dependency/'
    # analysis_dir = data_dir / 'analysis_dir/ohsumed_dependency/'
    # utils_dir = base_dir / 'utils/ohsumed_experiments'
    # experiment_name = 'ohsumed_4W_wA'
    # file_of_experiment = './data/ohsumed_dependency_clean2/'

    def __post_init__(self):
        # create the directory if they does not exist
        self.data_dir.mkdir(exist_ok=True)
        self.shuffle_index_dir.mkdir(exist_ok=True)
        self.vocab_path.mkdir(exist_ok=True)
        self.label_path.mkdir(exist_ok=True)
        self.graph_dir.mkdir(exist_ok=True)

        # self.checkpoint_dir.mkdir(exist_ok=True)
        # self.precess_dir.mkdir(exist_ok=True)
        # self.train_dir.mkdir(exist_ok=True)
        # self.analysis_dir.mkdir(exist_ok=True)
        # self.utils_dir.mkdir(exist_ok=True)


@dataclass
class GraphConfig:
    window_size = 20
    word_embeddings_dim = 300
    word_vector_file = 'data/glove.6B/glove.6B.200d.txt'

    def __post_init__(self):
        return


@dataclass
class TrainConfig:
    model = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
    learning_rate = 0.02  # Initial learning rate.
    epochs = 200  # Number of epochs to train.
    hidden1 = 200  # Number of units in hidden layer 1.
    dropout = 0.5  # Dropout rate (1 - keep probability).
    weight_decay = 0.  # Weight for L2 loss on embedding matrix.
    early_stopping = 10  # Tolerance for early stopping (# of epochs).
    max_degree = 3  # Maximum Chebyshev polynomial degree.

    def __post_init__(self):
        return


project = Project()
graph_config = GraphConfig()
train_config = TrainConfig()
