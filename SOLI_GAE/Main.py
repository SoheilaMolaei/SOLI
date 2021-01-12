import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import sys
from random import random
from nltk.cluster import em
import torch.nn as nn
import networkx as nx
from models import InfoPartitioned, LogReg
import process
from copy import deepcopy
from sklearn.decomposition import NMF
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import mask_test_edges,get_roc_score,acc_score
dataset = 'cora'


# training params
batch_size = 1
nb_epochs = 100
# nb_epochs = 1

patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
#middle_hid_unit
hid_units = 700

sparse = True
nonlinearity = 'prelu' # special name to separate parameters

adj, feature, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
X=feature.todense()
Y=labels
import networkx as nx
Inputs=deepcopy(X)
Input=deepcopy(Inputs)
dataMain=deepcopy(Input.astype(int))
dataMain=np.array(dataMain)
graphMain=nx.from_numpy_matrix(adj.todense())
listClique=list(nx.find_cliques(graphMain))

for i in listClique:
  for k,j in zip(i,i):
    if j!=k:
        Input[j]=Inputs[j]+ Inputs[k]
        Input[k]=Inputs[j]+ Inputs[k]

features=Input

nb_nodes = features.shape[0]
ft_size = features.shape[1]
n_clusters = labels.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])


model = InfoPartitioned(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()


b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]
    
    idxx = np.random.permutation(nb_nodes)
    shuf_ftss = features[:, idxx, :]
    
    idxxx = np.random.permutation(nb_nodes)
    shuf_ftsss = features[:, idxxx, :]
    
    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    

    logits = model(features, shuf_fts,shuf_ftss,shuf_ftsss, sp_adj if sparse else adj, sparse, None, None, None) 

    loss = b_xent(logits, lbl)
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()
model.load_state_dict(torch.load('best.pkl'))


embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
emd_lbls = torch.argmax(labels[0,:], dim=1)

embeds=torch.reshape(embeds,[embeds.shape[1],embeds.shape[2]])
from autoencoders import autoencoder,ClusteringLayer

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

x = embeds.cpu().detach().numpy()
y = emd_lbls.cpu().detach().numpy()

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(x)
print("Kmeans NMI",normalized_mutual_info_score(emd_lbls.cpu().detach().numpy(),y_pred_kmeans))
print("Kmeans ACC",acc_score(emd_lbls.cpu().detach().numpy(),y_pred_kmeans))

dims = [x.shape[-1], hid_units,32]

init = VarianceScaling(scale=1., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 300
batch_size = 64
save_dir = './results'
autoencoder, encoder = autoencoder(dims, init=init)
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.save_weights(save_dir + '/ae_weights2.h5')
autoencoder.load_weights(save_dir + '/ae_weights2.h5')

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model1 = Model(inputs=encoder.input, outputs=clustering_layer)
model1.compile(optimizer=SGD(0.01, 0.9), loss='kld')


kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred_last = np.copy(y_pred)
model1.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = 60
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.0001 # tolerance threshold to stop training
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model1.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(acc_score(y, y_pred), 5)
            nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(adjusted_rand_score(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model1.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model1.save_weights(save_dir + '/Soli.h5')
model1.load_weights(save_dir + '/Soli.h5')

q = model1.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(acc_score(y, y_pred), 5)
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
    
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

# plt.figure(figsize=(16, 14))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
# plt.title("Confusion matrix", fontsize=30)
# plt.ylabel('True label', fontsize=25)
# plt.xlabel('Clustering label', fontsize=25)
# plt.show()


roc_score, ap_score = get_roc_score(x, adj_orig, test_edges, test_edges_false)
tqdm.write('Test ROC score: ' + str(roc_score))
tqdm.write('Test AP score: ' + str(ap_score))

#Cora---ACC=0.765140, f1_macro=0.718944, precision_macro=0.804534, recall_macro=0.722651, f1_micro=0.765140, precision_micro=0.765140, recall_micro=0.765140, NMI=0.601065, ADJ_RAND_SCORE=0.566581
#Citeseer---ACC=0.671175, f1_macro=0.619219, precision_macro=0.640110, recall_macro=0.621541, f1_micro=0.671175, precision_micro=0.671175, recall_micro=0.671175, NMI=0.420832, ADJ_RAND_SCORE=0.399086
#pubmed ACC=0.670234, f1_macro=0.664024, precision_macro=0.687291, recall_macro=0.698920, f1_micro=0.670234, precision_micro=0.670234, recall_micro=0.670234, NMI=0.305828, ADJ_RAND_SCORE=0.282737
