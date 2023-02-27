import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from sklearn import preprocessing
import torch.autograd as autograd
from typing import Any
from torch.optim.lr_scheduler import LambdaLR

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

        # If current constituent has a closing parenthesis, process it
#         if constituency.find(')') > -1:
#             # Pop current constituent from stack
#             num = stack_num.pop()
#             const = stack_const.pop()

#             # If constituent is not ROOT, add it to word-to-constituents arcs
#             if const != "ROOT":
#                 if const not in w_c_to_idx:
#                     w_c_to_idx[const] = len(w_c_to_idx)
#                 word_to_constituents.append([w_c_to_idx[const], j, num, 1])

#             # If there are more constituents on the stack, add constituents-to-constituents arcs
#             if len(stack_num) != 0:
#                 # Add constituents-to-constituents arc from super to sub constituent
#                 if stack_const[-1] != "ROOT":
#                     if stack_const[-1] not in c_c_to_idx:
#                         c_c_to_idx[stack_const[-1]] = len(c_c_to_idx)
#                     constituents_to_constituents.append([c_c_to_idx[stack_const[-1]], stack_num[-1], num, 0])

#                     # Update children dictionary for super constituent
#                     if stack_const[-1] not in children:
#                         children[stack_const[-1]] = [const]
#                     else:
#                         children[stack_const[-1]].append(const)

#                 # Add constituents-to-constituents arc from sub to super constituent
#                 if const != "ROOT":
#                     if const not in c_c_to_idx:
#                         c_c_to_idx[const] = len(c_c_to_idx)
#                     constituents_to_constituents.append([c_c_to_idx[const], num, stack_num[-1], 1])
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
def read_data_pandas(df: pd.DataFrame, w_c_to_idx: dict, c_c_to_idx: dict, dep_lb_to_idx: dict) -> Tuple[List[List[Any]], dict, dict, dict]:
    """
    Reads in a pandas DataFrame containing data, and extracts features from the data using the `get_sentence_features`
    function. The features are stored in a list of lists, which is returned along with updated dictionaries containing
    word-to-index, constituent-to-index, and dependency label-to-index mappings.

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
    Reads in a pandas DataFrame containing data, and extracts features from the data using the `get_sentence_features`
    function. The features are stored in a list of lists, which is returned along with updated dictionaries containing
    word-to-index, constituent-to-index, and dependency label-to-index mappings. For SNLI, premise features are stored
    in a dict for efficiency.

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
# def read_dropna_encode_dataframe(file_name: str, le: preprocessing.LabelEncoder, fit: bool) -> pd.DataFrame:
#     """
#     Reads a pandas DataFrame from a pickle file, removes rows with NaN values and those with a gold_label of '-',
#     encodes the gold_label column using a LabelEncoder object and returns the resulting DataFrame.
    
#     Parameters:
#     -----------
#     file_name: str
#         The name of the pickle file to read the DataFrame from.
#     le: sklearn.preprocessing.LabelEncoder
#         A LabelEncoder object used to encode the gold_label column.
#     fit: bool
#         If True, fits the LabelEncoder on the gold_label column before encoding it. 
        
#     Returns:
#     --------
#     df: pd.DataFrame
#         The resulting DataFrame with the encoded gold_label column.
#     """
#     # read the DataFrame from the pickle file
#     df = pd.read_pickle(file_name)
    
#     # remove rows with NaN values and gold_label of '-'
#     df = df.dropna()
#     df = df[df.gold_label != '-']
    
#     # fit and transform the gold_label column if requested
#     if fit:
#         le.fit(list(df['gold_label']))
        
#     labels = encode_gold_labels(df, le)
    
#     # replace the original gold_label column with the encoded one
#     df['gold_label'] = labels.tolist()
    
#     # return the resulting DataFrame
#     return df
def read_dropna_encode_dataframe(file_name, le, fit, is_hans=False, is_mnli=False):
    """
    Reads a pandas dataframe from a pickle file, drops rows with missing values, and encodes the gold labels.
    
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


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
