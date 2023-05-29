# -*- coding: utf-8 -*-
"""CaHesyfu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xPgVxDzchv7ZBsKzJ4VoAdnp23q-tkNu

# Github Repo
https://github.com/lucalin17081994/Syntax-Enhanced-Bert

# Imports + GPU setup
"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install transformers
# !apt-get install git
# !git clone https://github.com/lucalin17081994/Syntax-Enhanced-Bert/
# %cd Syntax-Enhanced-Bert

import Modules
from Modules import CA_Hesyfu, initialize_model, initialize_model, WarmupLinearSchedule
import Data
from Data import (
    read_dropna_encode_dataframe,
    read_data_pandas_snli, 
    read_data_pandas, 
    SNLI_Dataset,
    get_batch_sup,
    dep_label_mappings,
    apply_clustering_dependency_labels
)

import Evaluation
from Evaluation import log_eval_metrics,train_batch
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
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from typing import Tuple, List, Callable, Optional
from torch import FloatTensor, LongTensor

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
   print("Training on GPU")
   device = torch.device("cuda:0")
else:
   print('training on CPU')
   device=torch.device("cpu")

from zipfile import ZipFile

file_name = "SNLI_2validations.zip"

with ZipFile(file_name, 'r') as zipfile:
    zipfile.printdir()
    zipfile.extractall()

# import pickle
# #hardcoding for ease
# dep_lb_to_idx= {'det': 0, 'nsubj': 1, 'case': 2, 'nmod': 3, 'root': 4, 'amod': 5, 'compound:prt': 6, 'obl': 7, 'punct': 8, 'aux': 9, 'nmod:poss': 10, 'obj': 11, 'cop': 12, 'advcl': 13, 'cc': 14, 'conj': 15, 'expl': 16, 'xcomp': 17, 'compound': 18, 'mark': 19, 'nummod': 20, 'advmod': 21, 'acl': 22, 'acl:relcl': 23, 'cc:preconj': 24, 'fixed': 25, 'nsubj:pass': 26, 'aux:pass': 27, 'ccomp': 28, 'appos': 29, 'iobj': 30, 'det:predet': 31, 'flat': 32, 'parataxis': 33, 'obl:tmod': 34, 'obl:agent': 35, 'discourse': 36, 'nmod:npmod': 37, 'goeswith': 38, 'csubj': 39, 'list': 40, 'obl:npmod': 41, 'dep': 42, 'reparandum': 43, 'nmod:tmod': 44, 'orphan': 45, 'dislocated': 46, 'vocative': 47, 'csubj:pass': 48}
# c_c_to_idx= {'NP': 0, 'DT': 1, 'NN': 2, 'PP': 3, 'IN': 4, 'S': 5, 'VP': 6, 'VBZ': 7, 'ADJP': 8, 'VBN': 9, 'RP': 10, '.': 11, 'VBG': 12, 'PRP$': 13, ',': 14, 'ADVP': 15, 'RB': 16, 'NNS': 17, 'CC': 18, 'PRP': 19, 'VBP': 20, 'EX': 21, 'JJ': 22, 'JJR': 23, 'SBAR': 24, 'TO': 25, 'VB': 26, 'CD': 27, 'HYPH': 28, 'WHNP': 29, 'WP': 30, 'VBD': 31, 'PRT': 32, 'NML': 33, 'NNP': 34, 'WHADVP': 35, 'WRB': 36, 'WDT': 37, 'POS': 38, 'QP': 39, 'WHADJP': 40, 'MD': 41, '``': 42, 'NNPS': 43, "''": 44, 'PDT': 45, 'RBR': 46, 'JJS': 47, 'UH': 48, 'UCP': 49, 'WHPP': 50, 'AFX': 51, 'SYM': 52, 'SINV': 53, 'X': 54, 'CONJP': 55, 'NFP': 56, 'FRAG': 57, 'WP$': 58, ':': 59, '-LRB-': 60, '-RRB-': 61, 'SBARQ': 62, 'SQ': 63, 'FW': 64, 'PRN': 65, 'INTJ': 66, 'RBS': 67, '$': 68, 'ADD': 69, 'RRC': 70, 'LST': 71, 'LS': 72, 'GW': 73}
# w_c_to_idx= {'S': 0, 'NP': 1, 'DT': 2, 'NN': 3, 'PP': 4, 'IN': 5, 'VP': 6, 'VBZ': 7, 'ADJP': 8, 'VBN': 9, 'RP': 10, '.': 11, 'VBG': 12, 'PRP$': 13, ',': 14, 'ADVP': 15, 'RB': 16, 'NNS': 17, 'CC': 18, 'PRP': 19, 'VBP': 20, 'EX': 21, 'JJ': 22, 'JJR': 23, 'SBAR': 24, 'TO': 25, 'VB': 26, 'CD': 27, 'HYPH': 28, 'WHNP': 29, 'WP': 30, 'VBD': 31, 'PRT': 32, 'NML': 33, 'NNP': 34, 'WHADVP': 35, 'WRB': 36, 'WDT': 37, 'POS': 38, 'QP': 39, 'WHADJP': 40, 'MD': 41, '``': 42, 'NNPS': 43, "''": 44, 'PDT': 45, 'RBR': 46, 'JJS': 47, 'UH': 48, 'UCP': 49, 'WHPP': 50, 'AFX': 51, 'SYM': 52, 'SINV': 53, 'X': 54, 'CONJP': 55, 'NFP': 56, 'FRAG': 57, 'WP$': 58, ':': 59, '-LRB-': 60, '-RRB-': 61, 'SBARQ': 62, 'SQ': 63, 'FW': 64, 'PRN': 65, 'INTJ': 66, 'RBS': 67, '$': 68, 'ADD': 69, 'RRC': 70, 'LST': 71, 'LS': 72, 'GW': 73}

le = preprocessing.LabelEncoder()

train_data=read_dropna_encode_dataframe('SNLI_train.pickle',le,True)
dev_data = read_dropna_encode_dataframe('SNLI_val.pickle',le,False)
dev_data2=read_dropna_encode_dataframe('SNLI_val_hard.pickle',le,False)

train_data=train_data.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)
dev_data=dev_data.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)
dev_data2=dev_data2.drop(['sentence1', 'sentence2', 'pos_sentence1', 'pos_sentence2'],axis=1)

# train_data = apply_clustering_dependency_labels(train_data, dep_label_mappings, -1)
# dev_data = apply_clustering_dependency_labels(dev_data, dep_label_mappings, -1)
# dev_data2 = apply_clustering_dependency_labels(dev_data2, dep_label_mappings, -1)

# help_data=read_dropna_encode_dataframe('HELP_train.pickle',le,False)
# help_data = help_data[['gold_label','text_sentence1','text_sentence2',	'heads_sentence1',	'heads_sentence2',	'deprel_sentence1',	'deprel_sentence2',	'sentence1_parse',	'sentence2_parse']]

train, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    train_data, {}, {}, {}, {}
)
print("train examples", len(train))

# help, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
#     help_data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict
# )
# print("help examples", len(help))


dev, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    dev_data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict
)
dev2, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict = read_data_pandas_snli(
    dev_data2, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict
)

print("dev examples", len(dev))
print("dev2 examples", len(dev2))

"""## Init model

model needs dependency vocab and constituency vocabs
"""

# use_constGCN and use_depGCN passed to initialize_model() and collate_fn
use_constGCN=True
use_depGCN=False
is_syntax_enhanced = use_constGCN or use_depGCN
model, model_name = initialize_model(768,1, dep_lb_to_idx,w_c_to_idx,c_c_to_idx,device, use_constGCN=use_constGCN, use_depGCN=use_depGCN)

"""## Init Dataloader"""

# # Create tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#re-seed. Init of model creates random parameters.
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Create train dataloader
train_dataset = SNLI_Dataset(train, tokenizer, premises_dict)
# help_dataset = SNLI_Dataset(help,tokenizer,premises_dict)
# combined = ConcatDataset([train_dataset,help_dataset])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

# Create validation dataloader
val_dataset = SNLI_Dataset(dev, tokenizer, premises_dict)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

# Create hard validation dataloader
val_dataset2 = SNLI_Dataset(dev2, tokenizer, premises_dict)
val_hard_dataloader = torch.utils.data.DataLoader(val_dataset2, batch_size=32, shuffle=False, collate_fn=lambda batch: get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN))

"""## wandb, hyperparameters, optimizers, schedulers"""

dataset_name = 'SNLI'
run_name = "SNLI_data_permutations"

# Hyperparameters
batch_size = train_dataloader.batch_size
n_epochs = 2
loss_fn = nn.CrossEntropyLoss()
learning_rate = 3e-5
lr_other = 5e-4

if is_syntax_enhanced:
    optimizer_bert = torch.optim.AdamW(model.bert.parameters(), lr=learning_rate, eps=1e-8)
    scheduler_bert = WarmupLinearSchedule(optimizer_bert,
                                          warmup_steps=int(len(train_dataloader) / 10),
                                          t_total=len(train_dataloader) * n_epochs)
    other_para = [
        {'params': model.hesyfu_layers.parameters()},
        {'params': model.co_attn.parameters()},
        {'params': model.fc.parameters()}
    ]
    optimizer_other = torch.optim.AdamW(other_para, lr=lr_other, eps=1e-8)
    scheduler_other = torch.optim.lr_scheduler.MultiStepLR(optimizer_other, milestones=[1], gamma=0.1, verbose=False)
else:     
    optimizer_bert = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler_bert = WarmupLinearSchedule(optimizer_bert,
                                          warmup_steps=int(len(train_dataloader) / 10),
                                          t_total=len(train_dataloader) * n_epochs)
    optimizer_other=None
# !pip install wandb -Uq
import wandb
wandb.login(key = 'a72edb442b6177a7198f045dee1e6b7c4de8f7a3')
run = wandb.init(project=run_name)
model_name = model_name
wandb.config.update({
    'model_name': model_name,
    'dataset_name': dataset_name,
    'lr_bert': learning_rate,
    'lr_other': lr_other,
    'epochs': n_epochs,
    'batch_size': batch_size,
    'loss_fn': loss_fn,
    'optimizer': optimizer_bert,
})
wandb.watch(model)

"""## training pipeline"""

train_losses, train_accuracies = [], []

# start training
for epc in range(n_epochs):
    model.to(device)
    print(f'start epoch {epc}')  
    
    # iterate through dataloader
    for i, batch in enumerate(train_dataloader):
        loss_batch, accuracy_batch = train_batch(model, batch, loss_fn, optimizer_bert, scheduler_bert, optimizer_other, device, is_syntax_enhanced)
        train_losses.append(loss_batch)
        train_accuracies.append(accuracy_batch)

        if i % 3000 == 0:
            print(f'evaluating batch nr:{i}')
            log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn, optimizer_bert, optimizer_other, device, wandb, is_syntax_enhanced)
            train_losses, train_accuracies = [], []
            
    if is_syntax_enhanced:
        scheduler_other.step()
    
#save model after last epoch
print('saving model')
model.cpu()
torch.save(model.state_dict(), model_name + '.pth')
artifact = wandb.Artifact(model_name + '.pth', type='model')
artifact.add_file(model_name + '.pth')
run.log_artifact(artifact)

print('last evaluation')
model.to(device)
log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn, optimizer_bert, optimizer_other, device, wandb,is_syntax_enhanced)
run.finish()
