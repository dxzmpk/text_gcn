from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn

import numpy as np     

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP

from Project import train_config



# if len(sys.argv) != 2:
# 	sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# dataset = sys.argv[1]
dataset = 'ohsumed'
if dataset not in datasets:
	sys.exit("wrong dataset name")
train_config.dataset = dataset

# Set random seed
seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    train_config.dataset)

features = sp.identity(features.shape[0])  # featureless


# Some preprocessing
features = preprocess_features(features)
if train_config.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif train_config.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, train_config.max_degree)
    num_supports = 1 + train_config.max_degree
    model_func = GCN
elif train_config.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(train_config.model))


# Define placeholders
t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_support = []
# for i in range(len(support)):
#     t_support.append(torch.Tensor(support[i]))
t_support.append(torch.Tensor(support[0]))
t_support_new = []
A_count = 4
count_accumulation = t_support[0]
for i in range(A_count-1):
    count_accumulation = count_accumulation.mm(t_support[0])
t_support_new.append(count_accumulation)
# t_support_new.append(t_support[0].mm(t_support[0]))
t_support = None
if torch.cuda.is_available() and False:
    t_features = t_features.cuda()
    t_y_train = t_y_train.cuda()
    t_y_val = t_y_val.cuda()
    t_y_test = t_y_test.cuda()
    t_train_mask = t_train_mask.cuda()
    tm_train_mask = tm_train_mask.cuda()
    for i in range(len(support)):
        t_support_new = [t.cuda() for t in t_support_new if True]
    model_func(input_dim=features.shape[0], support=t_support_new, num_classes=y_train.shape[1])
    model_func = model_func.cuda()
    print('Using CUDA.....')

else:
    model = model_func(input_dim=features.shape[0], support=t_support_new, num_classes=y_train.shape[1])
"""
注意！这里可能需要更改
"""

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)


# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)



val_losses = []

# Train model
for epoch in range(train_config.epochs):

    t = time.time()
    
    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item() # 正确的数目/总数目
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > train_config.early_stopping and val_losses[-1] > np.mean(val_losses[-(train_config.early_stopping + 1):-1]):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))


print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# doc and word embeddings
tmp = model.layer1.embedding.numpy()
word_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_doc_embeddings = tmp[:train_size]  # include val docs
test_doc_embeddings = tmp[adj.shape[0] - test_size:]

print('Embeddings:')
print('\rWord_embeddings:'+str(len(word_embeddings)))
print('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings))) 
print('\rTest_doc_embeddings:'+str(len(test_doc_embeddings))) 
print('\rWord_embeddings:') 
print(word_embeddings)

import pickle
with open(r'C:\全新的毕业设计\text_gcn.pytorch\data\r8_dependency\vocab_list.pkl', 'rb') as f:                     # open file with write-mode
    vocab_list =  pickle.load(f)

vocab_size = len(vocab_list)
word_vectors = []
for i in range(vocab_size):
    word = vocab_list[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open(project.file_of_experiment + dataset + '_word_vectors.txt', 'w') as f:
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
doc_vec_dir = os.path.join(project.train_dir,project.experiment_name+'.doc_vectors.txt')
with open(doc_vec_dir, 'w') as f:
    f.write(doc_embeddings_str)


