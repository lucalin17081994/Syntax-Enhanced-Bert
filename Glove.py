# -*- coding: utf-8 -*-
"""Glove.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xPgVxDzchv7ZBsKzJ4VoAdnp23q-tkNu

# Github Repo
https://github.com/lucalin17081994/Syntax-Enhanced-Bert

# Imports + GPU setup
"""



import Modules
from Modules import CA_Hesyfu, initialize_model, WarmupLinearSchedule
import Data
from Data import (
    read_dropna_encode_dataframe,
    read_data_pandas_snli, 
    read_data_pandas
)
import Evaluation
from Evaluation import log_eval_metrics,train_batch,log_eval_metrics_sick
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import time
import re
from sklearn import preprocessing
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple, List, Callable, Optional
from torch import FloatTensor, LongTensor



np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
   print("Training on GPU")
   device = torch.device("cuda:0")
else:
  device=torch.device("cpu")

"""# Run Pipeline

## Unzip file from drive
"""

# import pickle
# #hardcoding for ease
dep_lb_to_idx= {'det': 0, 'nsubj': 1, 'case': 2, 'nmod': 3, 'root': 4, 'amod': 5, 'compound:prt': 6, 'obl': 7, 'punct': 8, 'aux': 9, 'nmod:poss': 10, 'obj': 11, 'cop': 12, 'advcl': 13, 'cc': 14, 'conj': 15, 'expl': 16, 'xcomp': 17, 'compound': 18, 'mark': 19, 'nummod': 20, 'advmod': 21, 'acl': 22, 'acl:relcl': 23, 'cc:preconj': 24, 'fixed': 25, 'nsubj:pass': 26, 'aux:pass': 27, 'ccomp': 28, 'appos': 29, 'iobj': 30, 'det:predet': 31, 'flat': 32, 'parataxis': 33, 'obl:tmod': 34, 'obl:agent': 35, 'discourse': 36, 'nmod:npmod': 37, 'goeswith': 38, 'csubj': 39, 'list': 40, 'obl:npmod': 41, 'dep': 42, 'reparandum': 43, 'nmod:tmod': 44, 'orphan': 45, 'dislocated': 46, 'vocative': 47, 'csubj:pass': 48}
c_c_to_idx= {'NP': 0, 'DT': 1, 'NN': 2, 'PP': 3, 'IN': 4, 'S': 5, 'VP': 6, 'VBZ': 7, 'ADJP': 8, 'VBN': 9, 'RP': 10, '.': 11, 'VBG': 12, 'PRP$': 13, ',': 14, 'ADVP': 15, 'RB': 16, 'NNS': 17, 'CC': 18, 'PRP': 19, 'VBP': 20, 'EX': 21, 'JJ': 22, 'JJR': 23, 'SBAR': 24, 'TO': 25, 'VB': 26, 'CD': 27, 'HYPH': 28, 'WHNP': 29, 'WP': 30, 'VBD': 31, 'PRT': 32, 'NML': 33, 'NNP': 34, 'WHADVP': 35, 'WRB': 36, 'WDT': 37, 'POS': 38, 'QP': 39, 'WHADJP': 40, 'MD': 41, '``': 42, 'NNPS': 43, "''": 44, 'PDT': 45, 'RBR': 46, 'JJS': 47, 'UH': 48, 'UCP': 49, 'WHPP': 50, 'AFX': 51, 'SYM': 52, 'SINV': 53, 'X': 54, 'CONJP': 55, 'NFP': 56, 'FRAG': 57, 'WP$': 58, ':': 59, '-LRB-': 60, '-RRB-': 61, 'SBARQ': 62, 'SQ': 63, 'FW': 64, 'PRN': 65, 'INTJ': 66, 'RBS': 67, '$': 68, 'ADD': 69, 'RRC': 70, 'LST': 71, 'LS': 72, 'GW': 73}
w_c_to_idx= {'S': 0, 'NP': 1, 'DT': 2, 'NN': 3, 'PP': 4, 'IN': 5, 'VP': 6, 'VBZ': 7, 'ADJP': 8, 'VBN': 9, 'RP': 10, '.': 11, 'VBG': 12, 'PRP$': 13, ',': 14, 'ADVP': 15, 'RB': 16, 'NNS': 17, 'CC': 18, 'PRP': 19, 'VBP': 20, 'EX': 21, 'JJ': 22, 'JJR': 23, 'SBAR': 24, 'TO': 25, 'VB': 26, 'CD': 27, 'HYPH': 28, 'WHNP': 29, 'WP': 30, 'VBD': 31, 'PRT': 32, 'NML': 33, 'NNP': 34, 'WHADVP': 35, 'WRB': 36, 'WDT': 37, 'POS': 38, 'QP': 39, 'WHADJP': 40, 'MD': 41, '``': 42, 'NNPS': 43, "''": 44, 'PDT': 45, 'RBR': 46, 'JJS': 47, 'UH': 48, 'UCP': 49, 'WHPP': 50, 'AFX': 51, 'SYM': 52, 'SINV': 53, 'X': 54, 'CONJP': 55, 'NFP': 56, 'FRAG': 57, 'WP$': 58, ':': 59, '-LRB-': 60, '-RRB-': 61, 'SBARQ': 62, 'SQ': 63, 'FW': 64, 'PRN': 65, 'INTJ': 66, 'RBS': 67, '$': 68, 'ADD': 69, 'RRC': 70, 'LST': 71, 'LS': 72, 'GW': 73}

le = preprocessing.LabelEncoder()
le.fit(['contradiction', 'entailment', 'neutral']) #hardcode so you know the encoding in another notebook

train_data=read_dropna_encode_dataframe('SNLI_train.pickle',le,True)#.head(1000)
dev_data = read_dropna_encode_dataframe('SNLI_val.pickle',le,False)
dev_data2=read_dropna_encode_dataframe('SNLI_val_hard.pickle',le,False)
train_data=train_data.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)
dev_data=dev_data.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)
dev_data2=dev_data2.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)

train, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    train_data, {}, {}, {}, {}
)
print("train examples", len(train))

dev, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    dev_data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict
)
dev2, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    dev_data2, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict
)

print("dev examples", len(dev))
print("dev2 examples", len(dev2))





def load_glove_embeddings(file_path, embedding_dim):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            tmp =  line.split()
            word = tmp[0]
            vector = tmp[1:]
            embeddings[word]=vector
    return embeddings


glove_file_path = "glove.6B.100d.txt"  # Update this path to your local file
embedding_dim = 100
glove_embeddings = load_glove_embeddings(glove_file_path, embedding_dim)

embedding_dim = len(list(glove_embeddings.values())[0])
vocab_size = len(glove_embeddings) + 2 # Add 2 for the <UNK> and <PAD> tokens
word_to_index = {}
embedding_matrix = np.zeros((len(glove_embeddings) + 2, embedding_dim))
index_to_word = {}
for i, word in enumerate(['<PAD>', '<UNK>'] + list(glove_embeddings.keys())):
    if word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    word_to_index[word] = i
    index_to_word[i] = word  # Add this line



from Modules import Hesyfu, Attn, masked_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""# Glove model"""

class Glove_Hesyfu(nn.Module):
    def __init__(
        self,
        hidden_dim,
        L,
        dep_tag_vocab_size,
        w_c_vocab_size,
        c_c_vocab_size,
        device,
        glove,
        use_constGCN=True,
        use_depGCN=True,
        
    ):
        super(Glove_Hesyfu, self).__init__()

        self.device = device
        self.glove=glove
        self.vocab_size = glove.shape[0]  # Number of words in the vocabulary

        # self.embedding_layer = torch.nn.Embedding(self.vocab_size, glove.dim, padding_idx=glove.stoi['<PAD>'])
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, glove.shape[1], padding_idx=word_to_index['<PAD>'])
        self.embedding_layer.weight.data.copy_(torch.tensor(glove, dtype=torch.float))


        self.lstm = nn.LSTM(input_size=glove.shape[1], hidden_size=hidden_dim, num_layers=L, bidirectional=True, batch_first=True)
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

        self.lstm.apply(init_weights)
        # self.soft_attn=Soft_Attn(hidden_dim *2, hidden_dim *2)


        self.dep_tag_vocab_size = dep_tag_vocab_size
        self.w_c_vocab_size = w_c_vocab_size
        self.c_c_vocab_size = c_c_vocab_size
        
        self.dropout = nn.Dropout(p=0.1)
        self.embedding_dropout = nn.Dropout(p=0.2)
        
        self.use_constGCN=use_constGCN
        self.use_depGCN=use_depGCN

        if self.use_constGCN or self.use_depGCN:
            # Hesyfu
            self.hesyfu_layers = nn.ModuleList()
            for _ in range(1):
                hesyfu = Hesyfu(
                    hidden_dim*2,
                    dep_tag_vocab_size,
                    w_c_vocab_size,
                    c_c_vocab_size, 
                    use_constGCN,
                    use_depGCN,
                    self.device
                )
                self.hesyfu_layers.append(hesyfu)

        # Co-attention
        self.co_attn = Attn(hidden_dim*2, hidden_dim*2)
        
        self.fc = nn.Linear(hidden_dim * 4 * 2, 3)
    def forward(self, sentence1_data, sentence2_data, input_tensor1, input_tensor2):
        # Unpack data
        (mask_batch1, lengths_batch1, dependency_arcs1, dependency_labels1, constituent_labels1,
        const_GCN_w_c1, const_GCN_c_w1, const_GCN_c_c1, mask_const_batch1, plain_sentences1) = sentence1_data
        (mask_batch2, lengths_batch2, dependency_arcs2, dependency_labels2, constituent_labels2,
        const_GCN_w_c2, const_GCN_c_w2, const_GCN_c_c2, mask_const_batch2, plain_sentences2) = sentence2_data

        #word embeddings
        glove_embedding1 = self.embedding_layer(input_tensor1)
        glove_embedding2 = self.embedding_layer(input_tensor2)

        #pack
#         lengths_batch1 = lengths_batch1.cpu()
#         lengths_batch2 = lengths_batch2.cpu()
#         packed_embeddings1 = pack_padded_sequence(glove_embedding1, lengths_batch1, batch_first=True, enforce_sorted=False)
#         packed_embeddings2 = pack_padded_sequence(glove_embedding2, lengths_batch2, batch_first=True, enforce_sorted=False)
      
        
        # Pass sentences through GCN's
        # gcn_in1, gcn_in2 = embedded_input1,embedded_input2#self.linear(embedded_input1), self.linear(embedded_input2)

        #lstm
        lstm_out1, _ = self.lstm(glove_embedding1)
        lstm_out2, _ = self.lstm(glove_embedding2)
        
#         lstm_out1, _ = pad_packed_sequence(lstm_out1, batch_first=True)
#         lstm_out2, _ = pad_packed_sequence(lstm_out2, batch_first=True)
        
        if self.use_constGCN or self.use_depGCN:
            # gcn_in1, gcn_in2 = self.soft_attn(lstm_out1, lstm_out2, mask_batch1, mask_batch2)
            gcn_in1, gcn_in2 = lstm_out1,lstm_out2
            for hesyfu in self.hesyfu_layers:
                gcn_out1, gcn_out2 = hesyfu(gcn_in1, gcn_in2, sentence1_data, sentence2_data)
                gcn_in1, gcn_in2 = gcn_out1, gcn_out2
            co_attn_in1, co_attn_in2 = gcn_out1,gcn_out2
        else:
            co_attn_in1, co_attn_in2 = lstm_out1,lstm_out2
        # Pass sentences through co-attention layer

        data1, data2 = self.co_attn(co_attn_in1, co_attn_in2, mask_batch1, mask_batch2)
        # Create final representation
        final_representation = torch.cat((data1, data2, torch.abs(data1 - data2), torch.mul(data1, data2)), dim=1)
        out = self.fc(final_representation)

        return out
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
L=1
use_constGCN=False
use_depGCN=True
is_syntax_enhanced= True
hidden_dim=100

model = Glove_Hesyfu(hidden_dim, L, len(dep_lb_to_idx), len(w_c_to_idx), len(c_c_to_idx), device, embedding_matrix,
                  use_constGCN=use_constGCN, use_depGCN=use_depGCN)
from Modules import count_parameters
count_parameters(model)
model

"""# collate FN"""

from Data import get_const_adj_BE
def sentences_to_indices(sentences, word_to_index):
    batch_indices = []
    for sentence in sentences:
        words = [word.lower() for word in sentence]  # Lower case all words
        indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in words]
        batch_indices.append(indices)
    return batch_indices

def pad_sequences(batch_indices, padding_value=word_to_index['<PAD>']):
    max_length = max([len(indices) for indices in batch_indices])
    padded_batch = []
    for indices in batch_indices:
        padded_indices = indices + [padding_value] * (max_length - len(indices))
        padded_batch.append(padded_indices)
    return padded_batch

def get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN):
    
    # Extract sentence data from batch, 

    sentence1_data_indices = [1, 2, 3, 4, 5, 6]
    sentence1_batch_data = get_batch_sup_sentence(batch, sentence1_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN)
    sentence2_data_indices = [7, 8, 9, 10, 11, 12]
    sentence2_batch_data = get_batch_sup_sentence(batch, sentence2_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN)

    #labels
    labels_batch = torch.tensor([x[0] for x in batch], dtype=torch.float64, device=device)

    sentences = [sample[1] for sample in batch]
    batch_indices = sentences_to_indices(sentences, word_to_index)
    padded_batch = pad_sequences(batch_indices)
    input_tensor1 = torch.tensor(padded_batch, dtype=torch.long).to(device)
    sentences = [sample[7] for sample in batch]
    batch_indices = sentences_to_indices(sentences, word_to_index)
    padded_batch = pad_sequences(batch_indices)
    input_tensor2 = torch.tensor(padded_batch, dtype=torch.long).to(device)

    # Return the data
    return sentence1_batch_data, sentence2_batch_data, labels_batch, input_tensor1, input_tensor2
def get_batch_sup_sentence(batch, indices, device, dep_lb_to_idx, use_constGCN, use_depGCN):
    hidden_d = hidden_dim*2
    max_sent_len = max(len(d[indices[0]]) for d in batch)
    max_const_len = max(d[indices[5]] for d in batch)
    lengths = []
    batch_len = len(batch)

    #only create dep arcs and labels tensors if needed for depGCN
    dependency_arcs = torch.zeros((batch_len, max_sent_len, max_sent_len), requires_grad=False).to(device) if use_depGCN else None
    dependency_labels = torch.zeros((batch_len, max_sent_len), requires_grad=False, dtype=torch.long).to(device) if use_depGCN else None

    mask_batch = torch.zeros((batch_len, max_sent_len), requires_grad=False).to(device)
    bert_embs = torch.zeros((batch_len, max_sent_len, hidden_d), requires_grad=False).to(device)
    
    #only get constituent features if necessary
    constituent_labels = torch.zeros((batch_len, max_const_len, hidden_d), requires_grad=False).to(device) if use_constGCN else None
    const_mask = torch.zeros((batch_len, max_const_len), requires_grad=False).to(device) if use_constGCN else None
    plain_sentences = [d[indices[0]] for d in batch]


    for d, data in enumerate(batch):
        num_const = data[indices[5]]
        if use_constGCN:
            const_mask[d][:num_const] = 1.0

        for w, word in enumerate(data[indices[0]]):
            mask_batch[d, w] = 1.0

            if use_depGCN:
                dependency_labels[d, w] = dep_lb_to_idx[data[indices[2]][w]]
                dependency_arcs[d, w, w] = 1

                if data[indices[1]][w] != 0:
                    dep_head = data[indices[1]][w] - 1
                    dependency_arcs[d, w, dep_head] = 1
                    dependency_arcs[d, dep_head, w] = 1

        lengths.append(len(data[indices[0]]))
    
    if use_constGCN:
        batch_w_c = []
        for d in batch:
            batch_w_c.append([])
            for i in d[indices[3]]:
                batch_w_c[-1].append([])
                for j in i:
                    batch_w_c[-1][-1].append(j)

        batch_c_c = []
        for d in batch:
            batch_c_c.append([])
            for i in d[indices[3]]:
                batch_c_c[-1].append([])
                for j in i:
                    batch_c_c[-1][-1].append(j)

        for d, _ in enumerate(batch):
            for t, trip in enumerate(batch_w_c[d]):
                for e, elem in enumerate(trip):
                    if elem > 499:
                        batch_w_c[d][t][e] = (elem - 500) + max_sent_len

            for t, trip in enumerate(batch_c_c[d]):
                for e, elem in enumerate(trip):
                    if elem > 499:
                        batch_c_c[d][t][e] = (elem - 500) + max_sent_len

    const_GCN_w_c = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, 2, 2, forward=True, device=device
    ) if use_constGCN else None
    const_GCN_c_w = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, 5, 20, forward=False, device=device
    ) if use_constGCN else None
    const_GCN_c_c = get_const_adj_BE(
        batch_c_c, max_sent_len + max_const_len, 2, 7, forward=True, device=device
    ) if use_constGCN else None


    lengths_batch = torch.LongTensor(lengths).to(device)

    return [
        mask_batch,
        lengths_batch,
        dependency_arcs,
        dependency_labels,
        constituent_labels,
        const_GCN_w_c,
        const_GCN_c_w,
        const_GCN_c_c,
        const_mask,
        plain_sentences
    ]

"""## Init Dataloader"""

from Data import Glove_Dataset
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Create train dataloader
train_dataset = Glove_Dataset(train, premises_dict)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

# Create validation dataloader
val_dataset = Glove_Dataset(dev, premises_dict)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

# Create hard validation dataloader
val_dataset2 = Glove_Dataset(dev2, premises_dict)
val_hard_dataloader = torch.utils.data.DataLoader(val_dataset2, batch_size=32, shuffle=False, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

"""## wandb, hyperparameters, optimizers, schedulers"""
import wandb
wandb.login(key='a72edb442b6177a7198f045dee1e6b7c4de8f7a3')

dataset_name = 'SNLI'
run_name = "experiment_glove"
# is_syntax_enhanced=False
# Hyperparameters
batch_size = train_dataloader.batch_size
n_epochs = 4
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3

   

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
scheduler =  WarmupLinearSchedule(optimizer,
                                          warmup_steps=int(len(train_dataloader) / 10),
                                          t_total=len(train_dataloader) * n_epochs)


run = wandb.init(project=run_name)
model_name = 'Glove'
wandb.config.update({
    'model_name': model_name,
    'dataset_name': dataset_name,
    'lr_bert': learning_rate,
    'epochs': n_epochs,
    'batch_size': batch_size,
    'loss_fn': loss_fn,
    'optimizer': optimizer,
})
wandb.watch(model)

"""## training pipeline"""

def train_batch(model, data_batch, loss_fn, optimizer, device, is_syntax_enhanced):
    model.train()
    sentence1_data, sentence2_data, labels, input_tensor1, input_tensor2 = data_batch

    
    out=model(sentence1_data, sentence2_data, input_tensor1, input_tensor2)
    
    # Backward pass and optimization
    optimizer.zero_grad()

    loss = loss_fn(out, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    
    # Compute accuracy
    accuracy_batch = compute_accuracy_batch(out, labels)
    
    return loss.cpu().detach().numpy(), accuracy_batch

def eval_batch(model, data_batch, loss_fn, device, is_syntax_enhanced):
    sentence1_data, sentence2_data, labels, input_tensor1, input_tensor2 = data_batch
    out=model(sentence1_data, sentence2_data, input_tensor1, input_tensor2)
    loss = loss_fn(out, labels)
    accuracy_batch = compute_accuracy_batch(out, labels)
    return loss.item(), accuracy_batch
def eval_model(model, dataloader, loss_fn, device, is_syntax_enhanced):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in dataloader:
            loss_batch, accuracy_batch = eval_batch(model, batch, loss_fn, device, is_syntax_enhanced)
            losses.append(loss_batch)
            accuracies.append(accuracy_batch)
    return np.mean(losses), np.mean(accuracies)
def log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn, optimizer, device, wandb, is_syntax_enhanced):
    val_loss, val_accuracy = eval_model(model, val_dataloader, loss_fn, device, is_syntax_enhanced)
    val_loss_hard, val_accuracy_hard = eval_model(model, val_hard_dataloader, loss_fn, device, is_syntax_enhanced)
    
    wandb.log({
        'train_losses': np.mean(train_losses),
        'train_accuracies': np.mean(train_accuracies),
        'val_loss': val_loss.item(),
        'val_accuracy': val_accuracy.item(),
        'val_loss_hard': val_loss_hard,
        'val_acc_hard': val_accuracy_hard,
        'LR': optimizer.state_dict()['param_groups'][0]['lr'],
    })

from Evaluation import compute_accuracy_batch
train_losses, train_accuracies = [], []

# start training
for epc in range(n_epochs):
    model.to(device)
    print(f'start epoch {epc}')  
    
    # iterate through dataloader
    for i, batch in enumerate(train_dataloader):
        loss_batch, accuracy_batch = train_batch(model, batch, loss_fn, optimizer, device, is_syntax_enhanced)
        scheduler.step()
        train_losses.append(loss_batch)
        train_accuracies.append(accuracy_batch)

        if i % 1000 == 0:
            print(f'evaluating batch nr:{i}')
            log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn,optimizer, device, wandb, is_syntax_enhanced)
            train_losses, train_accuracies = [], []
    # scheduler.step()
    
#save model after last epoch
print('saving model')
model.cpu()
torch.save(model.state_dict(), model_name + '.pth')
artifact = wandb.Artifact(model_name + '.pth', type='model')
artifact.add_file(model_name + '.pth')
run.log_artifact(artifact)

print('last evaluation')
model.to(device)
log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn,optimizer, device, wandb, is_syntax_enhanced)
run.finish()

