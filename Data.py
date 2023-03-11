import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from sklearn import preprocessing
import torch.autograd as autograd
from typing import Any
from torch.utils.data import Dataset

def read_data_pandas(df: pd.DataFrame, w_c_to_idx: dict, c_c_to_idx: dict, dep_lb_to_idx: dict) -> Tuple[List[List[Any]], dict, dict, dict]:
    """
    Reads in a pandas DataFrame containing data, and extracts features from the data using the `get_sentence_features`
    function. The features are stored in a list of lists, which is returned along with updated dictionaries containing
    word-to-index, constituent-to-index, and dependency label-to-index mappings, which are the vocabs for dependency and constituency labels.

    Args:
        data_file (pandas.DataFrame): A pandas DataFrame containing the data to be processed.
        w_c_to_idx (Dict[str, int]): A dictionary mapping words to their corresponding indices.
        c_c_to_idx (Dict[str, int]): A dictionary mapping constituents to their corresponding indices.
        dep_lb_to_idx (Dict[str, int]): A dictionary mapping dependency labels to their corresponding indices.

    Returns:
        A tuple containing the processed data, updated word-to-index, constituent-to-index, and dependency-label-to-index mappings.
    """
    data = []
    
    for i in range(len(df)):
        
        # get the current line of df
        line = df.iloc[i]

        # get features for the first sentence
        text1, dep_head1, dep_lb1, word_to_constituents1, constituents_to_constituents1, curr_num1, \
            w_c_to_idx, c_c_to_idx, dep_lb_to_idx = get_sentence_features(
                line, 'text_sentence1', 'sentence1_parse', 'heads_sentence1', 'deprel_sentence1',
                w_c_to_idx, c_c_to_idx, dep_lb_to_idx)

        # get features for the second sentence
        text2, dep_head2, dep_lb2, word_to_constituents2, constituents_to_constituents2, curr_num2, \
            w_c_to_idx, c_c_to_idx, dep_lb_to_idx = get_sentence_features(
                line, 'text_sentence2', 'sentence2_parse', 'heads_sentence2', 'deprel_sentence2',
                w_c_to_idx, c_c_to_idx, dep_lb_to_idx)
        
        # get the gold label for the line
        label = line['gold_label']
      
        # add the line's data to the data list
        data.append([
            label, text1, dep_head1, dep_lb1, word_to_constituents1, constituents_to_constituents1,
            curr_num1 - 500, text2, dep_head2, dep_lb2, word_to_constituents2, constituents_to_constituents2,
            curr_num2 - 500])

    return data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx
def read_data_pandas_snli(df: pd.DataFrame, w_c_to_idx: dict, c_c_to_idx: dict, dep_lb_to_idx: dict, premises_dict: dict) -> Tuple[List[List[Any]], dict, dict, dict, dict]:
    """
    Optimization version of read_data_pandas. Premises occur 3-15x in SNLI, which makes it more efficient to store them only once in a dict.

    Args:
        data_file (pandas.DataFrame): A pandas DataFrame containing the data to be processed.
        w_c_to_idx (Dict[str, int]): A dictionary mapping words to their corresponding constituency indices.
        c_c_to_idx (Dict[str, int]): A dictionary mapping constituents to their corresponding indices.
        dep_lb_to_idx (Dict[str, int]): A dictionary mapping dependency labels to their corresponding indices.
        premises_dict (Dict[list, list]): A dictionary mapping sentences to features. Premises in SNLI are repeated 3x.

    Returns:
        A tuple containing the processed data, updated word-to-index, constituent-to-index, and dependency-label-to-index mappings.
    """
    data = []
    for i in range(len(df)):
        
        # get the current line of df
        line = df.iloc[i]

        # get features for the first sentence
        text1, dep_head1, dep_lb1, word_to_constituents1, constituents_to_constituents1, curr_num1, \
            w_c_to_idx, c_c_to_idx, dep_lb_to_idx = get_sentence_features(
                line, 'text_sentence1', 'sentence1_parse', 'heads_sentence1', 'deprel_sentence1',
                w_c_to_idx, c_c_to_idx, dep_lb_to_idx)

        # get features for the second sentence
        text2, dep_head2, dep_lb2, word_to_constituents2, constituents_to_constituents2, curr_num2, \
            w_c_to_idx, c_c_to_idx, dep_lb_to_idx = get_sentence_features(
                line, 'text_sentence2', 'sentence2_parse', 'heads_sentence2', 'deprel_sentence2',
                w_c_to_idx, c_c_to_idx, dep_lb_to_idx)
        
        # get the gold label for the line
        label = line['gold_label']

        #add premise to dictionary
        key=" ".join([x for x in text1])
        if key not in premises_dict:
            premises_dict[key] = [text1,
                                  dep_head1, 
                                  dep_lb1, 
                                  word_to_constituents1, 
                                  constituents_to_constituents1,
                                  curr_num1 - 500]
      
        # add the line's data to the data list
        data.append([
            label, 
            key,
            text2, dep_head2, dep_lb2, word_to_constituents2, constituents_to_constituents2,
            curr_num2 - 500])

    return data, w_c_to_idx, c_c_to_idx, dep_lb_to_idx, premises_dict

def get_sentence_features(line, text_name, parse_name, heads_name, deprel_name, w_c_to_idx, c_c_to_idx, dep_lb_to_idx):
    # Initialize variables
    dep_head = []
    dep_lb = []
    stack_const = []
    stack_num = []
    word_to_constituents = []
    constituents_to_constituents = []

    # Set initial value for curr_num
    curr_num = 500

    # Create dictionary to store children for each constituent
    children = {}

    # Extract text, POS, and parse tree from input line
    text = line[text_name]
    const_tree = line[parse_name]

    # Extract word-to-constituents and constituents-to-constituents arcs from parse tree
    for j, constituency in enumerate(const_tree.split("(")[1:]):
        # Extract constituent label
        const = constituency.split()[0] 

        # If constituent is not ROOT, add it to word-to-constituents arcs
        if const != "ROOT":
            if const not in w_c_to_idx:
                w_c_to_idx[const] = len(w_c_to_idx)
            word_to_constituents.append([w_c_to_idx[const], j, curr_num, 0]) # pos

        # Add current constituent number to stack
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
    
    # Extract dependency heads and labels from input line
    dep_head = line[heads_name]
    dep_lb = line[deprel_name]

    # Add new dependency labels to dictionary
    for d in dep_lb:
        if d not in dep_lb_to_idx.keys():
            dep_lb_to_idx[d] = len(dep_lb_to_idx)

    # Return relevant information
    return text, dep_head, dep_lb, word_to_constituents, constituents_to_constituents, curr_num, w_c_to_idx, c_c_to_idx, dep_lb_to_idx

def read_dropna_encode_dataframe(file_name, le, fit, is_hans=False, is_mnli=False):
    """
    Reads a pandas dataframe from a pickle file, drops rows with missing values, and encodes the gold labels in [x,y,z] format for crossentropy.
    
    Args:
    - file_name (str): name of the pickle file to read the dataframe from
    - le (sklearn.preprocessing.LabelEncoder): a label encoder object to encode the gold labels
    - fit (bool): whether to fit the label encoder to the gold labels
    - is_hans (bool): whether the gold labels should be converted to a binary classification of HANS format
    - is_mnli (bool): whether the gold labels should be converted to a three-class classification of MNLI format
    
    Returns:
    - df (pandas.DataFrame): the encoded dataframe
    
    """
    # Read the dataframe from the pickle file, drop rows with missing values, and filter out invalid gold labels
    df = pd.read_pickle(file_name).dropna().query('gold_label != "-"')

    # Fit the label encoder to the gold labels, if requested
    if fit:
        le.fit(df['gold_label'])

    # Convert the gold labels to binary classification of HANS format, if requested
    if is_hans:
        df['gold_label'] = df['gold_label'].map({1: 'neutral', 0: 'entailment'})

    # Convert the gold labels to three-class classification of MNLI format, if requested
    if is_mnli:
        df['gold_label'] = df['gold_label'].map({1: 'neutral', 0: 'entailment', 2: 'contradiction'})

    # Encode the gold labels using the label encoder and convert to list
    labels = encode_gold_labels(df, le)
    df['gold_label'] = labels.tolist()

    return df
def encode_gold_labels(df: pd.DataFrame, le: preprocessing.LabelEncoder) -> torch.Tensor:
    """
    Encodes the gold_label column of a pandas DataFrame using a LabelEncoder object and returns a one-hot encoded tensor.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to encode.
    le: sklearn.preprocessing.LabelEncoder
        A LabelEncoder object used to encode the gold_label column.
        
    Returns:
    --------
    encoded_labels: torch.Tensor
        A tensor containing the one-hot encoded gold_label column.
    """
    # get the gold_label column as a list
    gold_labels = list(df['gold_label'])
    
    # encode the gold_label column using the LabelEncoder object
    encoded_labels = le.transform(gold_labels)
    
    # convert the encoded gold_label column to a PyTorch tensor
    encoded_labels = torch.as_tensor(encoded_labels)
    
    # one-hot encode the tensor
    encoded_labels = F.one_hot(encoded_labels)
    
    # return the resulting tensor
    return encoded_labels


class UnevenLengthDataset(Dataset):
    '''
    Dataset object if you read the data without the premises_dictionary.
    '''
    def __init__(self, X, tokenizer) -> None:
        #tokenizer used in collate_fn in Dataloader
        
        self.Y=[x[0] for x in X]
        #sentence1 data
        self.text1=[x[1] for x in X]
        self.dep_head1=[x[2] for x in X]
        self.dep_lb1=[x[3] for x in X]
        self.word_to_constituents1=[x[4] for x in X]
        self.constituents_to_constituents1=[x[5] for x in X]
        self.number_constituents1=[x[6] for x in X]

        #sentence2 data
        self.text2=[x[7] for x in X]
        self.dep_head2=[x[8] for x in X]
        self.dep_lb2=[x[9] for x in X]
        self.word_to_constituents2=[x[10] for x in X]
        self.constituents_to_constituents2=[x[11] for x in X]
        self.number_constituents2=[x[12] for x in X]

         # Encoding sentences using BERT tokenizer
        self.bert_encoded_sentences = [
            tokenizer.encode(" ".join(sent1), " ".join(sent2), add_special_tokens=True)
            for sent1, sent2 in zip(self.text1, self.text2)
        ]
        
        # Converting encoded sentences to tokens using BERT tokenizer
        self.bert_tokenized_sentences = [
            tokenizer.convert_ids_to_tokens(bert_encoded_sentence)
            for bert_encoded_sentence in self.bert_encoded_sentences
        ]
        
    def __len__(self) -> int:
        return len(self.Y)
        
    def __getitem__(self, idx: int):

        
        return (
            self.Y[idx],
            #sentence1 data
            self.text1[idx],
            self.dep_head1[idx],
            self.dep_lb1[idx],
            self.word_to_constituents1[idx],
            self.constituents_to_constituents1[idx],
            self.number_constituents1[idx],
            #sentence2 data
            self.text2[idx],
            self.dep_head2[idx],
            self.dep_lb2[idx],
            self.word_to_constituents2[idx],
            self.constituents_to_constituents2[idx],
            self.number_constituents2[idx],
            self.bert_encoded_sentences[idx],
            self.bert_tokenized_sentences[idx],
            # idx
            
        )
class SNLI_Dataset(Dataset):
    '''
    Dataset with premise_dictionary functionality. 
    '''
    def __init__(self, X, tokenizer, premises_dict) -> None:
        # Extracting Y and sentence keys
        self.Y = [x[0] for x in X]
        keys = [x[1] for x in X]
        
        # Extracting sentence 1 features
        sentence1_features = [premises_dict[x] for x in keys]
        self.text1 = [x[0] for x in sentence1_features]
        self.dep_head1 = [x[1] for x in sentence1_features]
        self.dep_lb1 = [x[2] for x in sentence1_features]
        self.word_to_constituents1 = [x[3] for x in sentence1_features]
        self.constituents_to_constituents1 = [x[4] for x in sentence1_features]
        self.number_constituents1 = [x[5] for x in sentence1_features]

        # Extracting sentence 2 features
        self.text2 = [x[2] for x in X]
        self.dep_head2 = [x[3] for x in X]
        self.dep_lb2 = [x[4] for x in X]
        self.word_to_constituents2 = [x[5] for x in X]
        self.constituents_to_constituents2 = [x[6] for x in X]
        self.number_constituents2 = [x[7] for x in X]

        # Encoding sentences using BERT tokenizer
        self.bert_encoded_sentences = [
            tokenizer.encode(" ".join(sent1), " ".join(sent2), add_special_tokens=True)
            for sent1, sent2 in zip(self.text1, self.text2)
        ]
        
        # Converting encoded sentences to tokens using BERT tokenizer
        self.bert_tokenized_sentences = [
            tokenizer.convert_ids_to_tokens(bert_encoded_sentence)
            for bert_encoded_sentence in self.bert_encoded_sentences
        ]
        
    def __len__(self) -> int:
        return len(self.Y)
        
    def __getitem__(self, idx: int):
        # Returning all features for a given index
        return (
            self.Y[idx],
            # Sentence 1 features
            self.text1[idx],
            self.dep_head1[idx],
            self.dep_lb1[idx],
            self.word_to_constituents1[idx],
            self.constituents_to_constituents1[idx],
            self.number_constituents1[idx],

            # Sentence 2 features
            self.text2[idx],
            self.dep_head2[idx],
            self.dep_lb2[idx],
            self.word_to_constituents2[idx],
            self.constituents_to_constituents2[idx],
            self.number_constituents2[idx],
            self.bert_encoded_sentences[idx],
            self.bert_tokenized_sentences[idx]
        )

def get_batch_sup(batch, device, dep_lb_to_idx, use_constGCN, use_depGCN):
    
    # Extract sentence data from batch, 
    if use_constGCN or use_depGCN:
      sentence1_data_indices = [1, 2, 3, 4, 5, 6]
      sentence1_batch_data = get_batch_sup_sentence(batch, sentence1_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN)
      sentence2_data_indices = [7, 8, 9, 10, 11, 12]
      sentence2_batch_data = get_batch_sup_sentence(batch, sentence2_data_indices, device, dep_lb_to_idx, use_constGCN, use_depGCN)
    else:
      sentence1_batch_data = [None] * 10
      sentence2_batch_data = [None] * 10

    # Extract data for BERT
    labels_batch = torch.tensor([x[0] for x in batch], dtype=torch.float64, device=device)
    bert_encoded_sentences = [x[13] for x in batch]
    bert_tokenized_sentences = [x[14] for x in batch]

    # Pad input_ids and convert to tensors
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(encoded_sentence) for encoded_sentence in bert_encoded_sentences],
        padding_value=0,
        batch_first=True
    ).to(device)

    # Create attention mask and convert to tensors
    attention_mask = (input_ids != 0).float().to(device)

    # Return the data
    return sentence1_batch_data, sentence2_batch_data, labels_batch, input_ids, attention_mask, bert_tokenized_sentences
def get_batch_sup_sentence(batch, indices, device, dep_lb_to_idx, use_constGCN, use_depGCN):
    bert_hidden_dim = 768
    max_sent_len = max(len(d[indices[0]]) for d in batch)
    max_const_len = max(d[indices[5]] for d in batch)
    lengths = []
    batch_len = len(batch)

    #only create dep arcs and labels tensors if needed for depGCN
    dependency_arcs = torch.zeros((batch_len, max_sent_len, max_sent_len), requires_grad=False).to(device) if use_depGCN else None
    dependency_labels = torch.zeros((batch_len, max_sent_len), requires_grad=False, dtype=torch.long).to(device) if use_depGCN else None

    mask_batch = torch.zeros((batch_len, max_sent_len), requires_grad=False).to(device)
    bert_embs = torch.zeros((batch_len, max_sent_len, bert_hidden_dim), requires_grad=False).to(device)
    
    #only get constituent features if necessary
    constituent_labels = torch.zeros((batch_len, max_const_len, bert_hidden_dim), requires_grad=False).to(device) if use_constGCN else None
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

def get_const_adj_BE(batch, max_batch_len, max_degr_in, max_degr_out, forward,device):
    node1_index = [[word[1] for word in sent] for sent in batch]
    node2_index = [[word[2] for word in sent] for sent in batch]
    label_index = [[word[0] for word in sent] for sent in batch]
    begin_index = [[word[3] for word in sent] for sent in batch]

    batch_size = len(batch)

    _MAX_BATCH_LEN = max_batch_len
    _MAX_DEGREE_IN = max_degr_in
    _MAX_DEGREE_OUT = max_degr_out

    adj_arc_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 2), dtype="int32"
    )
    adj_lab_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 1), dtype="int32"
    )
    adj_arc_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 2), dtype="int32"
    )
    adj_lab_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 1), dtype="int32"
    )

    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN), dtype="float32")
    mask_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT), dtype="float32"
    )
    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype="float32")

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(node1_index):  # iterates over the batch
        for a, arc in enumerate(de):
            if not forward:
                arc_1 = arc
                arc_2 = node2_index[d][a]
            else:
                arc_2 = arc
                arc_1 = node2_index[d][a]

            if begin_index[d][a] == 0:  # BEGIN
                if arc_1 in tmp_in:
                    tmp_in[arc_1] += 1
                else:
                    tmp_in[arc_1] = 0

                idx_in = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_IN)
                    + arc_1 * _MAX_DEGREE_IN
                    + tmp_in[arc_1]
                )

                if tmp_in[arc_1] < _MAX_DEGREE_IN:
                    adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                    adj_lab_in[idx_in] = np.array([label_index[d][a]])  # incoming arcs
                    mask_in[idx_in] = 1.0

            else:  # END
                if arc_1 in tmp_out:
                    tmp_out[arc_1] += 1
                else:
                    tmp_out[arc_1] = 0

                idx_out = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_OUT)
                    + arc_1 * _MAX_DEGREE_OUT
                    + tmp_out[arc_1]
                )

                if tmp_out[arc_1] < _MAX_DEGREE_OUT:
                    adj_arc_out[idx_out] = np.array([d, arc_2])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array(
                        [label_index[d][a]]
                    )  # outgoing arcs
                    mask_out[idx_out] = 1.0

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = torch.LongTensor(np.transpose(adj_arc_in).tolist())
    adj_arc_out = torch.LongTensor(np.transpose(adj_arc_out).tolist())

    adj_lab_in = torch.LongTensor(np.transpose(adj_lab_in).tolist())
    adj_lab_out = torch.LongTensor(np.transpose(adj_lab_out).tolist())

    mask_in = autograd.Variable(
        torch.FloatTensor(
            mask_in.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_IN)).tolist()
        ),
        requires_grad=False,
    )
    mask_out = autograd.Variable(
        torch.FloatTensor(
            mask_out.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_OUT)).tolist()
        ),
        requires_grad=False,
    )
    mask_loop = autograd.Variable(
        torch.FloatTensor(mask_loop.tolist()), requires_grad=False
    )

    
    adj_arc_in = adj_arc_in.to(device)
    adj_arc_out = adj_arc_out.to(device)
    adj_lab_in = adj_lab_in.to(device)
    adj_lab_out = adj_lab_out.to(device)
    mask_in = mask_in.to(device)
    mask_out = mask_out.to(device)
    mask_loop = mask_loop.to(device)
    return [
        adj_arc_in,
        adj_arc_out,
        adj_lab_in,
        adj_lab_out,
        mask_in,
        mask_out,
        mask_loop,
    ]
def apply_clustering_dependency_labels(df, mapping, granularity):
    '''
    granularity should be either 0 or -1
    '''
    df['deprel_sentence1']=df['deprel_sentence1'].apply(lambda x: [mapping.get(i,i)[granularity] for i in x])
    df['deprel_sentence2']=df['deprel_sentence2'].apply(lambda x: [mapping.get(i,i)[granularity] for i in x])
    return df
