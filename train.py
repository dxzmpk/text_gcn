from __future__ import division
from __future__ import print_function

import time

import torch
import torch.nn as nn
from sklearn import metrics
from Project import *
from models.gcn import GCN
from utils import *


def trans2tensor(xy_info):
    features, y_train, y_val, y_test, train_mask, y_train = xy_info
    t_features = torch.from_numpy(features)
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
    return t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask


def evaluate(gcn, features, labels, mask, criterion):
    t_test = time.time()
    gcn.eval()
    with torch.no_grad():
        out = gcn(features)
        t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(out * tm_mask, torch.max(labels, 1)[1])
        prediction = torch.max(out, 1)[1]
        acc = ((prediction == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, prediction.numpy(), labels.numpy(), (time.time() - t_test)


def print_test_result(test_mask, prediction, labels):
    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(prediction[i])
            test_labels.append(np.argmax(labels[i]))

    print_log("Test Precision, Recall and F1-Score...")
    print_log(metrics.classification_report(test_labels, test_pred, digits=4))
    print_log("Macro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print_log("Micro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


def train(gcn, xy_info, val_mask, criterion):
    val_losses = []
    t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask = \
        trans2tensor(xy_info)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=train_config.learning_rate)

    for epoch in range(train_config.epochs):
        t = time.time()
        # Forward pass
        out = gcn(t_features)
        loss = criterion(out * tm_train_mask, torch.max(t_y_train, 1)[1])
        acc = ((torch.max(out, 1)[1] == torch.max(t_y_train, 1)[1])
               .float() * t_train_mask).sum().item() / t_train_mask.sum().item()  # 正确的数目/总数目
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Validation
        val_loss, val_acc, pred, labels, duration = evaluate(gcn, t_features, t_y_val, val_mask, criterion)
        val_losses.append(val_loss)

        print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, "
                  "time= {:.5f}".format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

        if epoch > train_config.early_stopping and val_losses[-1] > np.mean(
                val_losses[-(train_config.early_stopping + 1):-1]):
            print_log("Early stopping...")
            break
    print_log("Optimization Finished!")
    return t_features, t_y_test


# doc and word embeddings
def store_word_doc_vectors(gcn, train_size, test_size, adj):
    tmp = gcn.layer1.embedding.numpy()
    word_embeddings = tmp[train_size: adj.shape[0] - test_size]
    train_doc_embeddings = tmp[:train_size]  # include val docs
    test_doc_embeddings = tmp[adj.shape[0] - test_size:]

    with open(project.vocab_path / (project.dataset + '.txt'), 'r') as f:
        words = f.readlines()

    vocab_size = len(words)
    word_vectors = []
    for i in range(vocab_size):
        word = words[i].strip()
        word_vector = word_embeddings[i]
        word_vector_str = ' '.join([str(x) for x in word_vector])
        word_vectors.append(word + ' ' + word_vector_str)
    word_embeddings_str = '\n'.join(word_vectors)
    with open(project.experiment_dir / 'word_vectors.txt', 'w') as f:
        f.write(word_embeddings_str)

    doc_vectors = []
    doc_id = 0
    for i in range(train_size):
        doc_vector = train_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    for i in range(test_size):
        doc_vector = test_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    doc_embeddings_str = '\n'.join(doc_vectors)
    with open(project.experiment_dir / 'doc_vectors.txt', 'w') as f:
        f.write(doc_embeddings_str)


def main():
    seed = 2021
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    )

    features = sp.identity(features.shape[0])

    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    gcn = GCN(input_dim=features.shape[0], support=torch.tensor(support, dtype=torch.float32),
              num_classes=y_train.shape[1])

    criterion = nn.CrossEntropyLoss()
    t_features, t_y_test = train(gcn, [features, y_train, y_val, y_test, train_mask, y_train], val_mask, criterion)

    test_loss, test_acc, pred, labels, test_duration = evaluate(gcn, t_features, t_y_test, test_mask, criterion)
    print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}"
              .format(test_loss, test_acc, test_duration))
    print_test_result(test_mask, pred, labels)
    store_word_doc_vectors(gcn, train_size, test_size, adj)


if __name__ == '__main__':
    main()
