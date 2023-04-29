from Modules import Hesyfu, Attn, masked_softmax
from Modules import count_parameters
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from Evaluation import compute_accuracy_batch


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
        word_to_index,
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
        final_representation = torch.cat((data1, data2, data1 - data2, torch.mul(data1, data2)), dim=1)
        out = self.fc(final_representation)

        return out
        
def init_glove_model(hidden_dim, L, len_dep_lb_to_idx, len_w_c_to_idx, len_c_c_to_idx, device, embedding_matrix, word_to_index,
                  use_constGCN, use_depGCN):
    model = Glove_Hesyfu(hidden_dim, L, len_dep_lb_to_idx, len_w_c_to_idx, len_c_c_to_idx, device, embedding_matrix, word_to_index,
                  use_constGCN=use_constGCN, use_depGCN=use_depGCN)
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
def get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN, hidden_dim, word_to_index):
    
    # Extract sentence data from batch, 

    sentence1_data_indices = [1, 2, 3, 4, 5, 6]
    sentence1_batch_data = get_batch_sup_sentence(batch, sentence1_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN, hidden_dim)
    sentence2_data_indices = [7, 8, 9, 10, 11, 12]
    sentence2_batch_data = get_batch_sup_sentence(batch, sentence2_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN, hidden_dim)

    #labels
    labels_batch = torch.tensor([x[0] for x in batch], dtype=torch.float64, device=device)

    sentences = [sample[1] for sample in batch]
    batch_indices = sentences_to_indices(sentences, word_to_index)
    padded_batch = pad_sequences(batch_indices, word_to_index['<PAD>'])
    input_tensor1 = torch.tensor(padded_batch, dtype=torch.long).to(device)
    sentences = [sample[7] for sample in batch]
    batch_indices = sentences_to_indices(sentences, word_to_index)
    padded_batch = pad_sequences(batch_indices,word_to_index['<PAD>'])
    input_tensor2 = torch.tensor(padded_batch, dtype=torch.long).to(device)

    # Return the data
    return sentence1_batch_data, sentence2_batch_data, labels_batch, input_tensor1, input_tensor2
def get_batch_sup_sentence(batch, indices, device, dep_lb_to_idx, use_constGCN, use_depGCN, hidden_dim):
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
def eval_batch(model, data_batch, loss_fn, device):
    sentence1_data, sentence2_data, labels, input_tensor1, input_tensor2 = data_batch
    out=model(sentence1_data, sentence2_data, input_tensor1, input_tensor2)
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
def log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn, optimizer, device, wandb):
    val_loss, val_accuracy = eval_model(model, val_dataloader, loss_fn, device)
    val_loss_hard, val_accuracy_hard = eval_model(model, val_hard_dataloader, loss_fn, device)
    
    wandb.log({
        'train_losses': np.mean(train_losses),
        'train_accuracies': np.mean(train_accuracies),
        'val_loss': val_loss.item(),
        'val_accuracy': val_accuracy.item(),
        'val_loss_hard': val_loss_hard,
        'val_acc_hard': val_accuracy_hard,
        'LR': optimizer.state_dict()['param_groups'][0]['lr'],
    })
