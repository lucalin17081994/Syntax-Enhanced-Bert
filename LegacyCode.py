class CA_Hesyfu(nn.Module):
  def __init__(
      self,
      hidden_dim,
      L,
      dep_tag_vocab_size,
      w_c_vocab_size,
      c_c_vocab_size,  
      device,
      use_constGCN = True,
      use_depGCN = True
      
  ):
    super(CA_Hesyfu, self).__init__()
    self.device=device
    self.dep_tag_vocab_size=dep_tag_vocab_size
    self.w_c_vocab_size=w_c_vocab_size
    self.c_c_vocab_size=c_c_vocab_size
    # self.vocab_size = w_c_vocab_size
    self.dropout = nn.Dropout(p=0.1)
    self.embedding_dropout = nn.Dropout(p=0.2)
    #bert

    # self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    self.bert = BertModel.from_pretrained("bert-base-uncased")

    #self.tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # self.bert=DistilBertModel.from_pretrained("distilbert-base-uncased")
    #hesyfu
    self.hesyfu_layers=nn.ModuleList()
    for i in range(L):

      hesyfu = Hesyfu(
        hidden_dim,
        dep_tag_vocab_size,
        w_c_vocab_size,
        c_c_vocab_size, 
        use_constGCN,
        use_depGCN,
        self.device
      )
      self.hesyfu_layers.append(hesyfu)

    

    #CA
    self.co_attn = Attn(768,768)
    
    self.fc = nn.Linear(768*4,3) #embedding x4 because final representation is a concat
    
  def forward(self, sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences):

    #unpack data
    mask_batch1,\
    lengths_batch1,\
    dependency_arcs1,\
    dependency_labels1,\
    constituent_labels1,\
    const_GCN_w_c1,\
    const_GCN_c_w1,\
    const_GCN_c_c1,\
    mask_const_batch1, \
    plain_sentences1 = sentence1_data

    mask_batch2,\
    lengths_batch2,\
    dependency_arcs2,\
    dependency_labels2,\
    constituent_labels2,\
    const_GCN_w_c2,\
    const_GCN_c_w2,\
    const_GCN_c_c2,\
    mask_const_batch2 ,\
    plain_sentences2= sentence2_data

    #pass sentences through bert
    bert_embs1, bert_embs2 = \
        self.get_bert_output(lengths_batch1,lengths_batch2,
                             plain_sentences1,plain_sentences2,
                             input_ids, attention_mask, bert_tokenized_sentences)

    #pass sentences through GCN's
    #TODO: confirm these are padded
    gcn_in1,gcn_in2 = bert_embs1,bert_embs2
    for hesyfu in self.hesyfu_layers:
      
      gcn_out1,gcn_out2 = hesyfu(
        gcn_in1, gcn_in2,\
        sentence1_data,sentence2_data
      )
      gcn_in1,gcn_in2 = gcn_out1,gcn_out2
    # gcn_out1,gcn_out2 = self.hesyfu2(
    #   gcn_out1, gcn_out2,\
    #   sentence1_data,sentence2_data
    # )

    #pass sentences through co-attention layer. First add+norm
    # co_attn_in1 = self.layernorm(gcn_out1 + bert_embs1)
    # co_attn_in2= self.layernorm(gcn_out2 + bert_embs2)
    co_attn_in1 = gcn_out1
    co_attn_in2= gcn_out2
    
    data1, data2 = self.co_attn(co_attn_in1,co_attn_in2,mask_batch1,mask_batch2)
    
    #create final representation 
    #final_representation=torch.cat([h_A,h_B, h_A-h_B , h_A*h_B],-1)
    final_representation = torch.cat((data1, data2, torch.abs(data1 - data2), torch.mul(data1, data2)), 1)
    out = self.fc(final_representation)

    return out

  '''
  Go transform embedding from bert to stanza tokens. Mean pooling
  '''
  def _find_corr_features(self, output_features, stanza_tokens, bert_tokens):
      curr_pos = 0
      token_target = ""
      idx_list = []
      feature_tensor = []

      flag = True
      #n_bert_subtokens=0
      #iterate over bert tokens
      for idx, token in enumerate(bert_tokens):
          if token.lower() == stanza_tokens[curr_pos].lower() \
                  or (len(token) == len(stanza_tokens[curr_pos].lower()) and flag == True) \
                  or token == "[UNK]" :
              curr_pos += 1
              idx_list.append([idx])
              token_target = ""
              continue
          
          #token is bert subtoken
          elif token.startswith('##'): #bert subtoken
              #n_bert_subtokens+=1
              token_target += token.lstrip('#')
              idx_list[-1].append(idx)
          else:
              token_target += token
              if flag:
                  idx_list.append([idx])
                  flag = False
              else:
                  idx_list[-1].append([idx])

          if token_target == stanza_tokens[curr_pos].lower() or len(token_target) == len(stanza_tokens[curr_pos].lower()):
              curr_pos += 1
              token_target = ""
              flag = True
      merge_type='mean'
      for sub_idx_list in idx_list:
          flattened_sub_idx_list = list(flatten_recursive(sub_idx_list))
          if merge_type == 'mean':
              sub_feature = torch.mean(output_features[flattened_sub_idx_list[:], :], dim=0, keepdim=False)
          else:
              sub_feature = output_features[flattened_sub_idx_list[0], :]

          sub_feature = sub_feature.unsqueeze(dim=0)
          if len(feature_tensor) > 0:
              feature_tensor = torch.cat((feature_tensor, sub_feature), 0)
          else:
              feature_tensor = sub_feature

      return feature_tensor
  '''
  process plain sentences using bert tokenizer,
  pass them through bert to get output of final layer,
  and remove all subtoken embeddings to make compatible with GCN's
  '''
  def get_bert_output(self, lengths1, lengths2, plain_sentences1,plain_sentences2, input_ids, attention_mask, bert_tokenized_sentences):

    bert_last_vectors = self.bert(input_ids, attention_mask=attention_mask)[0]
    batch_len= bert_last_vectors.shape[0]
    bert_last_vectors = bert_last_vectors.to(device) 
    max_seq_len1 = torch.max(lengths1)
    max_seq_len2 = torch.max(lengths2)
    #print(sentences_batch1.shape[1], sentences_batch2.shape[1])
    bert_embs1 = torch.zeros((batch_len, max_seq_len1, 768)).to(device)
    bert_embs2 = torch.zeros((batch_len, max_seq_len2, 768)).to(device)
    
    for s, sent in enumerate(bert_last_vectors):

        sep1_pos = bert_tokenized_sentences[s].index("[SEP]")
        
        text1_feature = self._find_corr_features(sent[1:sep1_pos, :], 
                                                 plain_sentences1[s], 
                                                 bert_tokenized_sentences[s][1:sep1_pos],
                                        )
        text2_feature = self._find_corr_features(sent[sep1_pos+1:-1, :], 
                                                 plain_sentences2[s], 
                                                 bert_tokenized_sentences[s][sep1_pos+1:-1],
                                        )

        seq_len1 = text1_feature.shape[0]
        seq_len2 = text2_feature.shape[0]
        bert_embs1[s, 0:seq_len1,:]=text1_feature
        bert_embs2[s, 0:seq_len2,:]=text2_feature
    
    return bert_embs1,bert_embs2
'''
abstraction to avoid repeating code for sentence 2

Parameters:
text_name, number_constituents_name, i_text_name, lower_text_name, 
dep_lb_name, dep_head_name, 
word_to_constituents_name
          : String, name of corresponding col of pandas df
'''
def get_batch_sup_sentence(
    batch, bert_hidden_dim,device,
    text_name, number_constituents_name, i_text_name, lower_text_name, 
    dep_lb_name, dep_head_name, 
    word_to_constituents_name
):
    max_sent_len = 0
    max_const_len = 0
    lengths = []
    for d in batch:
        max_sent_len = max(len(d[i_text_name]), max_sent_len)
        max_const_len = max(d[number_constituents_name], max_const_len)

    batch_len = len(batch)
    dependency_arcs = np.zeros((batch_len, max_sent_len, max_sent_len))
    dependency_labels = np.zeros((batch_len, max_sent_len))
    mask = np.zeros((batch_len, max_sent_len))
    bert_embs = np.zeros((batch_len, max_sent_len, 768))
    constituent_labels = np.zeros((batch_len, max_const_len, bert_hidden_dim))
    const_mask = np.zeros((batch_len, max_const_len))
    plain_sentences = []
    
    for d, data in enumerate(batch):

        num_const = data[number_constituents_name]
        const_mask[d][:num_const] = 1.0
        for w, word in enumerate(data[i_text_name]):

            mask[d, w] = 1.0
            word_lower = data[lower_text_name][w]
            dependency_labels[d, w] = dep_lb_to_idx[data[dep_lb_name][w]]
            dependency_arcs[d, w, w] = 1
            # ignore 0 index because it does not point to anything.
            # heads in dep parsing using stanza refer to id, not position, so do -1.

            if data[dep_head_name][w] != 0:
              dependency_arcs[d, w, data[dep_head_name][w]-1] = 1
              dependency_arcs[d, data[dep_head_name][w]-1, w] = 1

        plain_sentences.append(data[text_name])
        lengths.append(len(data[i_text_name]))

    batch_w_c = []
    for d in batch:
        batch_w_c.append([])
        for i in d[word_to_constituents_name]:
            batch_w_c[-1].append([])
            for j in i:
                batch_w_c[-1][-1].append(j)

    batch_c_c = []
    for d in batch:
        batch_c_c.append([])
        for i in d[word_to_constituents_name]:
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
    )
    const_GCN_c_w = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, 5, 20, forward=False, device=device
    )

    const_GCN_c_c = get_const_adj_BE(
        batch_c_c, max_sent_len + max_const_len, 2, 7, forward=True, device=device
    )
    mask_batch = _make_VariableFloat(mask, device, False)
    mask_const_batch = _make_VariableFloat(const_mask, device, False)
    lengths_batch = _make_VariableLong(lengths, device, False)
    constituent_labels = _make_VariableFloat(constituent_labels, device, False)
    dependency_labels = _make_VariableLong(dependency_labels, device, False)
    dependency_arcs = _make_VariableFloat(dependency_arcs, device, False)
    return (
        [mask_batch,
        lengths_batch,
        dependency_arcs,
        dependency_labels,
        constituent_labels,
        const_GCN_w_c,
        const_GCN_c_w,
        const_GCN_c_c,
        mask_const_batch,
        plain_sentences]
    )

'''
Code used for batching, from original code. 
Probably better to convert to a torch dataset and dataloader.
'''
def get_batch_sup(batch, bert_hidden_dim, device):

    sentence1_batch_data=get_batch_sup_sentence(
        batch, bert_hidden_dim,device,
        "text1","number_constituents1","i_text1","lower_text1", 
        "dep_lb1", "dep_head1",
        "word_to_constituents1"
    )

    sentence2_batch_data=get_batch_sup_sentence(
        batch, bert_hidden_dim,device,
        "text2","number_constituents2","i_text2","lower_text2", 
        "dep_lb2", "dep_head2",
        "word_to_constituents2"
    )
    labels_batch = torch.tensor([i['label'] for i in batch], requires_grad=False, dtype=torch.float64).to(device)

    return sentence1_batch_data, sentence2_batch_data, labels_batch
def get_sentence_features(line, text_name, pos_name, parse_name, heads_name, deprel_name, w_c_to_idx, c_c_to_idx, dep_lb_to_idx):
    
    # Initialize variables
    dep_head = []
    dep_lb = []
    stack_const = []
    stack_num = []
    word_to_constituents = []
    constituents_to_constituents = []


    curr_num = 500
    children = {} #store children of constituents

    text=line[text_name]
    pos=line[pos_name]

    const_tree = line[parse_name]
    #adds the words to constituents arcs

    for j, constituency in enumerate(const_tree.split("(")[1:]):
      const=constituency.split()[0] 
      if const == "ROOT":
        pass
      else:
        if const not in w_c_to_idx:
          w_c_to_idx[const] = len(w_c_to_idx)
        word_to_constituents.append(
          [w_c_to_idx[const], j, curr_num, 0] # pos
        )

      stack_num.append(curr_num)
      stack_const.append(const)
      curr_num += 1

      if constituency.find(')') >-1:
        for c in constituency:
          if c == ")":
            num = stack_num.pop()
            const = stack_const.pop()
            if const == "ROOT":
              pass
            else:
           
              if const not in w_c_to_idx:
                w_c_to_idx[const] = len(w_c_to_idx)
              word_to_constituents.append(
                [w_c_to_idx[const], j, num, 1] 
              )
          

            if len(stack_num) != 0:

              if stack_const[-1] == "ROOT":
                pass
              else:

                if stack_const[-1] not in c_c_to_idx:
                  c_c_to_idx[stack_const[-1]] = len(c_c_to_idx)
                constituents_to_constituents.append(
                  [
                    c_c_to_idx[stack_const[-1]],
                    stack_num[-1],
                    num,
                    0,
                  ]
                )  # from super to sub

                if stack_const[-1] not in children:
                  children[stack_const[-1]] = [const]
                else:
                  children[stack_const[-1]].append(const)

              if const == "ROOT":
                pass
              else:
                if const not in c_c_to_idx:
                    c_c_to_idx[const] = len(c_c_to_idx)
                constituents_to_constituents.append(
                    [c_c_to_idx[const], num, stack_num[-1], 1]
                )
    
    dep_head = line[heads_name]
    dep_lb=line[deprel_name]
    #assert(len(text)==len(dep_head))
    for d in dep_lb:
      if d not in dep_lb_to_idx.keys():
          dep_lb_to_idx[d] = len(dep_lb_to_idx)

    return text, pos, dep_head, dep_lb, word_to_constituents, constituents_to_constituents, curr_num, w_c_to_idx, c_c_to_idx, dep_lb_to_idx





def read_data_pandas(data_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx):
  

    data, curr_sent = [],[]

    for i in range(len(data_file)):
      line = data_file.iloc[i]

      
      text1, dep_head1, dep_lb1, word_to_constituents1, constituents_to_constituents1, curr_num1, w_c_to_idx, c_c_to_idx, dep_lb_to_idx\
      = get_sentence_features(line, 'text_sentence1', 'sentence1_parse','heads_sentence1','deprel_sentence1', w_c_to_idx, c_c_to_idx, dep_lb_to_idx)
        
      text2, dep_head2, dep_lb2, word_to_constituents2, constituents_to_constituents2, curr_num2, w_c_to_idx, c_c_to_idx, dep_lb_to_idx\
      = get_sentence_features(line, 'text_sentence2', 'sentence2_parse','heads_sentence2','deprel_sentence2', w_c_to_idx, c_c_to_idx, dep_lb_to_idx)
        
      label=line['gold_label']
      
      data.append([label,
              text1,
              dep_head1,
              dep_lb1,
              word_to_constituents1,
              constituents_to_constituents1,
              curr_num1 - 500,
              text2,
              dep_head2,
              dep_lb2,
              word_to_constituents2,
              constituents_to_constituents2,
              curr_num2 - 500]
      )


    return data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx

def train_batch(model, data_batch, loss_fn, optimizer,scheduler,optimizer_other,device):
  model.train()

  #unpack data
  sentence1_data, sentence2_data, labels, input_ids, attention_mask, bert_tokenized_sentences=data_batch
  out = model(sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences) #forward pass

  optimizer.zero_grad()
  optimizer_other.zero_grad()

  loss= loss_fn(out,labels)
  loss.backward()

  torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

  optimizer.step()
  scheduler.step()
  optimizer_other.step()
  # scheduler_other.step()  
  accuracy_batch = compute_accuracy_batch(out,labels)

  return loss.cpu().detach().numpy(), accuracy_batch  


def compute_accuracy_batch(prediction, target):
  y_pred=torch.argmax(prediction,dim=1)
  y_target=torch.argmax(target,dim=1)
  batch_size=len(y_target)
  return torch.sum(y_pred == y_target).item()/batch_size

def get_batch_sup_sentence(
    batch, indices
):
    bert_hidden_dim=768
    max_sent_len = 0
    max_const_len = 0
    lengths = []
    for d in batch:
        max_sent_len = max(len(d[indices[0]]), max_sent_len)
        max_const_len = max(d[indices[5]], max_const_len)

    batch_len = len(batch)
    
    dependency_arcs = torch.zeros((batch_len, max_sent_len, max_sent_len),requires_grad=False).to(device)
    dependency_labels = torch.zeros((batch_len, max_sent_len),requires_grad=False, dtype=torch.long).to(device)
    mask_batch = torch.zeros((batch_len, max_sent_len),requires_grad=False).to(device)
    bert_embs = torch.zeros((batch_len, max_sent_len, bert_hidden_dim),requires_grad=False).to(device)
    constituent_labels = torch.zeros((batch_len, max_const_len, bert_hidden_dim),requires_grad=False).to(device)
    const_mask = torch.zeros((batch_len, max_const_len),requires_grad=False).to(device)
    plain_sentences = []
    
    for d, data in enumerate(batch):

        num_const = data[indices[5]]
        const_mask[d][:num_const] = 1.0
        for w, word in enumerate(data[indices[0]]):

            mask_batch[d, w] = 1.0
            #word_lower = data[indices[7]][w]
            dependency_labels[d, w] = dep_lb_to_idx[data[indices[2]][w]]
            
            dependency_arcs[d, w, w] = 1
            #DONE: try using original code
            # ignore 0 index because it does not point to anything.
            # heads in dep parsing using stanza refer to id, not position, so do -1.
            if data[indices[1]][w] != 0:
              dependency_arcs[d, w, data[indices[1]][w]-1] = 1
              dependency_arcs[d, data[indices[1]][w]-1, w] = 1
            #below original code. Performs much worse because last index replaced with dep head so loss of info
            # dependency_arcs[d, w, data[indices[1]][w]-1] = 1
            # dependency_arcs[d, data[indices[1]][w]-1, w] = 1

        plain_sentences.append(data[indices[0]])
        lengths.append(len(data[indices[0]]))

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
    )
    const_GCN_c_w = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, 5, 20, forward=False, device=device
    )

    const_GCN_c_c = get_const_adj_BE(
        batch_c_c, max_sent_len + max_const_len, 2, 7, forward=True, device=device
    )
    # const_GCN_w_c = None
    # const_GCN_c_w = None
    # const_GCN_c_c = None

    lengths_batch = _make_VariableLong(lengths, device, False)

    return (
        [mask_batch,
        lengths_batch,
        dependency_arcs,
        dependency_labels,
        constituent_labels,
        const_GCN_w_c,
        const_GCN_c_w,
        const_GCN_c_c,
        const_mask,
        plain_sentences]
    )



def get_batch_sup(batch):
    
    labels_batch = torch.tensor([x[0] for x in batch], requires_grad=False, dtype=torch.float64).to(device)

    sentence1_data_indices = [1,2,3,4,5,6]
    sentence1_batch_data=get_batch_sup_sentence(
        batch, sentence1_data_indices
    )
    sentence2_data_indices=[7,8,9,10,11,12]
    sentence2_batch_data=get_batch_sup_sentence(
        batch, sentence2_data_indices
    )

    bert_encoded_sentences = [x[13] for x in batch]
    bert_tokenized_sentences= [x[14] for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [
            torch.tensor(bert_encoded_sentence)
            for bert_encoded_sentence in bert_encoded_sentences
        ],
        padding_value=0,
        batch_first=True,
    )
    input_ids = input_ids.to(device)
    attention_mask = (input_ids != 0).float().to(device)
    return sentence1_batch_data, sentence2_batch_data, labels_batch, input_ids, attention_mask, bert_tokenized_sentences
