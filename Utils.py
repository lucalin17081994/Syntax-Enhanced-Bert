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
        if constituency.find(')') > -1:
            # Pop current constituent from stack
            num = stack_num.pop()
            const = stack_const.pop()

            # If constituent is not ROOT, add it to word-to-constituents arcs
            if const != "ROOT":
                if const not in w_c_to_idx:
                    w_c_to_idx[const] = len(w_c_to_idx)
                word_to_constituents.append([w_c_to_idx[const], j, num, 1])

            # If there are more constituents on the stack, add constituents-to-constituents arcs
            if len(stack_num) != 0:
                # Add constituents-to-constituents arc from super to sub constituent
                if stack_const[-1] != "ROOT":
                    if stack_const[-1] not in c_c_to_idx:
                        c_c_to_idx[stack_const[-1]] = len(c_c_to_idx)
                    constituents_to_constituents.append([c_c_to_idx[stack_const[-1]], stack_num[-1], num, 0])

                    # Update children dictionary for super constituent
                    if stack_const[-1] not in children:
                        children[stack_const[-1]] = [const]
                    else:
                        children[stack_const[-1]].append(const)

                # Add constituents-to-constituents arc from sub to super constituent
                if const != "ROOT":
                    if const not in c_c_to_idx:
                        c_c_to_idx[const] = len(c_c_to_idx)
                    constituents_to_constituents.append([c_c_to_idx[const], num, stack_num[-1], 1])

    # Extract dependency heads and labels from input line
    dep_head = line[heads_name]
    dep_lb = line[deprel_name]

    # Add new dependency labels to dictionary
    for d in dep_lb:
        if d not in dep_lb_to_idx.keys():
            dep_lb_to_idx[d] = len(dep_lb_to_idx)

    # Return relevant information
    return text, dep_head, dep_lb, word_to_constituents, constituents_to_constituents, curr_num, w_c_to_idx, c_c_to_idx, dep_lb_to_idx
def read_data_pandas(df: pd.DataFrame, w_c_to_idx: dict, c_c_to_idx: dict, dep_lb_to_idx: dict, premises_dict: dict) -> Tuple[List[List[Any]], dict, dict, dict, dict]:
    """
    Reads in a pandas DataFrame containing data, and extracts features from the data using the `get_sentence_features`
    function. The features are stored in a list of lists, which is returned along with updated dictionaries containing
    word-to-index, constituent-to-index, and dependency label-to-index mappings.

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
import torch.autograd as autograd
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
  def read_dropna_encode_dataframe(file_name: str, le: preprocessing.LabelEncoder, fit: bool) -> pd.DataFrame:
    """
    Reads a pandas DataFrame from a pickle file, removes rows with NaN values and those with a gold_label of '-',
    encodes the gold_label column using a LabelEncoder object and returns the resulting DataFrame.
    
    Parameters:
    -----------
    file_name: str
        The name of the pickle file to read the DataFrame from.
    le: sklearn.preprocessing.LabelEncoder
        A LabelEncoder object used to encode the gold_label column.
    fit: bool
        If True, fits the LabelEncoder on the gold_label column before encoding it. 
        
    Returns:
    --------
    df: pd.DataFrame
        The resulting DataFrame with the encoded gold_label column.
    """
    # read the DataFrame from the pickle file
    df = pd.read_pickle(file_name)
    
    # remove rows with NaN values and gold_label of '-'
    df = df.dropna()
    df = df[df.gold_label != '-']
    
    # fit and transform the gold_label column if requested
    if fit:
        le.fit(list(df['gold_label']))
        
    labels = encode_gold_labels(df, le)
    
    # replace the original gold_label column with the encoded one
    df['gold_label'] = labels.tolist()
    
    # return the resulting DataFrame
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
  def compute_accuracy_batch(prediction, target):
    y_pred = torch.argmax(prediction, dim=1)
    y_target = torch.argmax(target, dim=1)
    return (y_pred == y_target).float().mean().item()

def train_batch(model, data_batch, loss_fn, optimizer, scheduler, optimizer_other, device):
    model.train()
    
    # Unpack data
    sentence1_data, sentence2_data, labels, input_ids, attention_mask, bert_tokenized_sentences = data_batch
    
    # Forward pass
    out = model(sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    optimizer_other.zero_grad()
    loss = loss_fn(out, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    scheduler.step()
    optimizer_other.step()
    # scheduler_other.step()
    
    # Compute accuracy
    accuracy_batch = compute_accuracy_batch(out, labels)
    
    return loss.cpu().detach().numpy(), accuracy_batch

def eval_batch(model, data_batch, loss_fn, device):
    sentence1_data, sentence2_data, labels, input_ids, attention_mask, bert_tokenized_sentences = data_batch
    out = model(sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences)
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
