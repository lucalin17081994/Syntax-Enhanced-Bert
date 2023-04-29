'''
NOTEBOOK LINK:
https://colab.research.google.com/drive/1JIa6kwBOf52GbHurKwoBt3e6Uc1E-BkQ#scrollTo=azKgay74du92
'''

from Modules import Hesyfu, Attn, masked_softmax
from Modules import count_parameters
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from Evaluation import compute_accuracy_batch


class Glove_HOB(nn.Module):
    def __init__(
        self,
        hidden_dim,
        L,
        device,
        glove,
        word_to_index,
    ):
        super(Glove_HOB, self).__init__()

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

        
        self.fc = nn.Linear(hidden_dim * 2, 3)
    def forward(self, input_tensor2):

        #word embeddings, 2nd sentence is hypothesis, so keep 2
        glove_embedding2 = self.embedding_layer(input_tensor2)

        #lstm
        lstm_out2, _ = self.lstm(glove_embedding2)
        
        out = self.fc(lstm_out2)
        return out
        
def init_glove_HOB(hidden_dim, L, device, embedding_matrix, word_to_index):
    model = Glove_HOB(hidden_dim, L, device, embedding_matrix, word_to_index)
    params = count_parameters(model)
    return model, params

from Data import get_const_adj_BE
def sentences_to_indices(sentences, word_to_index):
    batch_indices = []
    for sentence in sentences:
        words = [word.lower() for word in sentence]  # Lower case all words
        indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in words]
        batch_indices.append(indices)
    return batch_indices

def pad_sequences(batch_indices, padding_value):
    max_length = max([len(indices) for indices in batch_indices])
    padded_batch = []
    for indices in batch_indices:
        padded_indices = indices + [padding_value] * (max_length - len(indices))
        padded_batch.append(padded_indices)
    return padded_batch

def load_glove_embeddings_and_mappings(file_path, embedding_dim):
    # Load GloVe embeddings from file
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            tmp = line.split()
            word = tmp[0]
            vector = tmp[1:]
            embeddings[word] = vector

    # Create an embedding matrix and initialize it with GloVe embeddings
    embedding_dim = len(list(embeddings.values())[0])
    vocab_size = len(embeddings) + 2  # Add 2 for the <UNK> and <PAD> tokens
    word_to_index = {}
    embedding_matrix = np.zeros((len(embeddings) + 2, embedding_dim))
    index_to_word = {}
    for i, word in enumerate(['<PAD>', '<UNK>'] + list(embeddings.keys())):
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
        word_to_index[word] = i
        index_to_word[i] = word

    return word_to_index, index_to_word, embedding_matrix
def train_batch(model, data_batch, loss_fn, optimizer, device):
    model.train()
    labels, input_tensor2, indices = data_batch
    
    out=model(input_tensor2)
    
    # Backward pass and optimization
    optimizer.zero_grad()

    loss = loss_fn(out, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    
    # Compute accuracy
    accuracy_batch = compute_accuracy_batch(out, labels)
    
    return loss.cpu().detach().numpy(), accuracy_batch
def eval_batch(model, data_batch, loss_fn, device):
    labels, input_tensor2, indices = data_batch
    out=model(input_tensor2)
    loss = loss_fn(out, labels)
    accuracy_batch = compute_accuracy_batch(out, labels)
    return loss.item(), accuracy_batch
def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in dataloader:
            loss_batch, accuracy_batch = eval_batch(model, batch, loss_fn, device)
            losses.append(loss_batch)
            accuracies.append(accuracy_batch)
    return np.mean(losses), np.mean(accuracies)
def log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, loss_fn, optimizer, device, wandb):
    val_loss, val_accuracy = eval_model(model, val_dataloader, loss_fn, device)
    
    wandb.log({
        'train_losses': np.mean(train_losses),
        'train_accuracies': np.mean(train_accuracies),
        'val_loss': val_loss.item(),
        'val_accuracy': val_accuracy.item(),
        'val_loss_hard': val_loss_hard,
        'LR': optimizer.state_dict()['param_groups'][0]['lr'],
    })
    
def get_batch_sup_HOB(batch, device, hidden_dim, word_to_index):

    #labels
    labels_batch = torch.tensor([x[0] for x in batch], dtype=torch.float64, device=device)

    sentences = [sample[1] for sample in batch]
    batch_indices = sentences_to_indices(sentences, word_to_index)
    padded_batch = pad_sequences(batch_indices,word_to_index['<PAD>'])
    input_tensor2 = torch.tensor(padded_batch, dtype=torch.long).to(device)
    indices = torch.tensor([x[2] for x in batch], dtype=torch.int64, device=device)
    # Return the data
    return labels_batch, input_tensor2, indices
