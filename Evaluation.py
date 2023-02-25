import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle



def get_hans_results(model_name, file_name, preds, true):
    """
    Computes HANS results for a given model's predictions and true labels.

    Args:
        model_name: A string containing the name of the model.
        file_name: A string containing the file name of the HANS dataset.
        preds: A list of predicted labels.
        true: A list of true labels.

    Returns:
        A list containing the model name and the HANS results.
    """
    # Read HANS dataset
    df_hans = pd.read_pickle(file_name)

    # Add predicted and true labels to HANS dataset
    df_hans['preds'] = preds
    df_hans['true_label'] = true

    # Get accuracy for each heuristic and subcase
    accuracies = []
    for label in [1,2]:
        for heuristic in ['lexical_overlap', 'subsequence', 'constituent']:
            df_subcase = df_hans[(df_hans['heuristic'] == heuristic) & (df_hans['true_label'] == label)]
            accuracy = get_accuracy(df_subcase['preds'], df_subcase['true_label'])
            accuracies.append(accuracy)

    # Compute overall HANS accuracy, entailment accuracy, and neutral accuracy
    hans_accuracy = get_accuracy(preds, true)
    entailment_accuracy = get_accuracy(df_hans[df_hans['true_label'] == 1]['preds'], df_hans[df_hans['true_label'] == 1]['true_label'])
    neutral_accuracy = get_accuracy(df_hans[df_hans['true_label'] == 2]['preds'], df_hans[df_hans['true_label'] == 2]['true_label'])

    # Return results as a list
    results = [model_name] + accuracies + [entailment_accuracy, neutral_accuracy, hans_accuracy]
    return results
def get_med_results(model_name, file_name, preds, true):
    df_med = pd.read_csv(file_name, sep='\t', on_bad_lines='skip')
    df_med['preds'] = preds
    df_med['true_label'] = true

    results = [
        model_name,
        get_accuracy(df_med[df_med['genre'].str.contains('upward_monotone')]['preds'], df_med[df_med['genre'].str.contains('upward_monotone')]['true_label']),
        get_accuracy(df_med[df_med['genre'].str.contains('downward_monotone')]['preds'], df_med[df_med['genre'].str.contains('downward_monotone')]['true_label']),
        get_accuracy(df_med[df_med['genre'].str.contains('non_monotone')]['preds'], df_med[df_med['genre'].str.contains('non_monotone')]['true_label']),
        get_accuracy(df_med['preds'], df_med['true_label'])
    ]
    return results
def med_results_linguistic_phenomena(model_name, df, preds, true):
    df['preds'] = preds
    df['true_label'] = true

    df_med_upward = df[df['genre'].str.contains('upward_monotone')]
    df_med_downward = df[df['genre'].str.contains('downward_monotone')]
    df_med_nonmonotone = df[df['genre'].str.contains('non_monotone')]
    
    results = [model_name] + [item for sublist in [analyze_upward_cases(df_med_upward), 
                                                   analyze_downward_cases(df_med_downward), 
                                                   analyze_non_monotone_cases(df_med_nonmonotone)] 
                              for item in sublist]
    
    return results
def get_accuracy(preds, true):
    print(f"preds: {preds} ({type(preds)}), true: {true} ({type(true)})")
    return ((np.asarray(preds) == np.asarray(true)).mean() * 100)
# def get_accuracy(preds, true):
#   preds=np.asarray(preds)
#   true=np.asarray(true)
#   return (preds==true).mean()*100

def get_preds_from_logits(logits, add_contradiction_and_neutral):

    if add_contradiction_and_neutral:
        m = nn.Softmax(dim=0)
        flattened = [m(item) for sublist in logits for item in sublist]
        flattened = [torch.tensor([x.item(),(y+z).item()]) for y,x,z in flattened] #add 0 and 2 index
        preds = [torch.argmax(x).tolist()+1 for x in flattened] #+1 because contradiction merged with neutral so entailment became 0, non-entailment became 1
    else:
        preds = [torch.argmax(x, dim=1).tolist() for x in logits]
        preds = [item for sublist in preds for item in sublist]
    return preds
def pickle_predictions(file_name, preds):
    with open(file_name, "wb") as fp:   #Pickling
        pickle.dump(preds, fp)
def get_pickled_predictions(file_name):
    with open(file_name, "rb") as fp:   # Unpickling
        return pickle.load(fp)

def analyze_upward_cases(df):
    accuracies = []
    lexical=df[df['genre'].str.contains('lexical', case=False)]
    non_lexical = df[~df['genre'].str.contains('lexical', case=False)]
    accuracies.append(get_accuracy(lexical['preds'],lexical['true_label']))
    accuracies.append(get_accuracy(non_lexical['preds'],non_lexical['true_label']))
    genres = ['npi', 'conditionals', 'conjunction', 'disjunction', 'reverse']
    accuracies.append([get_accuracy(df[df['genre'].str.contains(genre, case=False)]['preds'],
                              df[df['genre'].str.contains(genre, case=False)]['true_label'])
                for genre in genres])

    return accuracies
def analyze_downward_cases(df):  
    accuracies = []
    lexical=df[df['genre'].str.contains('lexical', case=False)]
    non_lexical = df[~df['genre'].str.contains('lexical', case=False)]
    accuracies.append(get_accuracy(lexical['preds'],lexical['true_label']))
    accuracies.append(get_accuracy(non_lexical['preds'],non_lexical['true_label']))
    genres = ['npi', 'conditionals', 'conjunction', 'disjunction']
    accuracies.append([get_accuracy(df[df['genre'].str.contains(genre, case=False)]['preds'],
                              df[df['genre'].str.contains(genre, case=False)]['true_label'])
                for genre in genres])

    return accuracies

def analyze_non_monotone_cases(df):
  
    lexical=df[df['genre'].str.contains('lexical', case=False)]
    non_lexical = df[~df['genre'].str.contains('lexical', case=False)]
    npi= df[df['genre'].str.contains('npi', case=False)]
    disjunction = df[df['genre'].str.contains('disjunction', case=False)]

    accuracy_lexical= get_accuracy(lexical['preds'],lexical['true_label'])
    accuracy_non_lexical=get_accuracy(non_lexical['preds'],non_lexical['true_label'])
    accuracy_npi = get_accuracy(npi['preds'],npi['true_label'])
    accuracy_disjunction = get_accuracy(disjunction['preds'],disjunction['true_label'])
    return [accuracy_lexical,accuracy_non_lexical,accuracy_npi,accuracy_disjunction]
def med_results_linguistic_phenomena(model_name, df, preds, true):
    df['preds']=preds
    df['true_label']=true

    df_med_upward = df[df['genre'].str.contains('upward_monotone')]
    df_med_downward=df[df['genre'].str.contains('downward_monotone')]
    df_med_nonmonotone = df[df['genre'].str.contains('non_monotone')]
    results=[]
    results.append(analyze_upward_cases(df_med_upward))
    results.append(analyze_downward_cases(df_med_downward))
    results.append(analyze_non_monotone_cases(df_med_nonmonotone))
    results = [item for sublist in results for item in sublist]
    results.insert(0, model_name)
    return results
def get_model_results_snli_mnli(model_name, true_snli_test, true_snli_test_hard, true_mnli_m, true_mnli_mm):
    results = [model_name]
    results.append(get_accuracy(get_pickled_predictions(model_name + '_preds_snli_test.pickle'), true_snli_test))
    results.append(get_accuracy(get_pickled_predictions(model_name + '_preds_snli_test_hard.pickle'), true_snli_test_hard))
    results.append(get_accuracy(get_pickled_predictions(model_name + '_preds_mnli_m.pickle'), true_mnli_m))
    results.append(get_accuracy(get_pickled_predictions(model_name + '_preds_mnli_mm.pickle'), true_mnli_mm))
    return results
def get_hans_results_subcases(file_name):
    # Load the HANS validation dataset
    df_hans = pd.read_pickle('HANS_val_original.pickle')
    # Load the predicted labels
    df_hans['preds'] = get_pickled_predictions(file_name)
    # Add the true labels to the dataframe
    df_hans['true_label'] = true_hans

    # Group the HANS dataset by heuristic and subcase
    heuristics_subcases = df_hans.groupby('heuristic')['subcase'].unique()

    # Get separate dataframes for entailment and neutral examples
    df_hans_entailment = df_hans[df_hans['true_label'] ==1]
    df_hans_neutral = df_hans[df_hans['true_label'] ==2]

    # Initialize lists to store the results for entailment and neutral examples
    results_entailment, results_nonentailment = [],[]

    # Loop over the subcases of each heuristic
    for heuristic in heuristics_subcases:
        for subcase in heuristic:
            subcase_list = [subcase]
            # Get the subset of the dataset corresponding to the current subcase and entailment
            df_subcase_entailment = df_hans_entailment[df_hans_entailment['subcase'] == subcase]
            # Get the subset of the dataset corresponding to the current subcase and neutral
            df_subcase_nonentailment = df_hans_neutral[df_hans_neutral['subcase'] == subcase]
            # If there are any examples in the entailment subset, compute the accuracy and add it to the results
            if len(df_subcase_entailment) > 0:
                preds = df_subcase_entailment['preds']
                true = df_subcase_entailment['true_label']
                accuracy_subcase = get_accuracy(preds, true)
                subcase_list.append(accuracy_subcase)
                results_entailment.append(subcase_list)
            # If there are any examples in the neutral subset, compute the accuracy and add it to the results
            if len(df_subcase_nonentailment) > 0:
                preds = df_subcase_nonentailment['preds']
                true = df_subcase_nonentailment['true_label']
                accuracy_subcase = get_accuracy(preds, true)
                subcase_list.append(accuracy_subcase)
                results_nonentailment.append(subcase_list)
    # Return the results for entailment and neutral examples as separate lists
    return results_entailment, results_nonentailment
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

def log_eval_metrics(model, train_losses, train_accuracies, val_dataloader, val_hard_dataloader, loss_fn, optimizer_bert, optimizer_other, device, wandb):
    val_loss, val_accuracy = eval_model(model, val_dataloader, loss_fn, device)
    val_loss_hard, val_accuracy_hard = eval_model(model, val_hard_dataloader, loss_fn, device)
    wandb.log({
        'train_losses': np.mean(train_losses),
        'train_accuracies': np.mean(train_accuracies),
        'val_loss': val_loss.item(),
        'val_accuracy': val_accuracy.item(),
        'val_loss_hard': val_loss_hard,
        'val_acc_hard': val_accuracy_hard,
        'LR_bert': optimizer_bert.state_dict()['param_groups'][0]['lr'],
        'LR_others': optimizer_other.state_dict()['param_groups'][0]['lr']
    })

def eval_batch_store_preds(model, data_batch, loss_fn, device, is_syntax_enhanced):
    """
    Evaluates a batch of data and stores the model's predictions.

    Args:
        model: A PyTorch model to evaluate.
        data_batch: A tuple containing the data batch to evaluate.
        loss_fn: A PyTorch loss function to use for evaluation.
        device: The device to use for evaluation (e.g., "cpu" or "cuda").
        is_syntax_enhanced: A boolean indicating whether to use the syntax-enhanced model.

    Returns:
        A tuple containing the loss, accuracy, and model predictions.
    """
    # Unpack data batch
    sentence1_data, sentence2_data, labels, input_ids, attention_mask, bert_tokenized_sentences = data_batch

    # Call model with appropriate arguments
    if is_syntax_enhanced:
        out = model(sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences)
    else:
        out = model(input_ids, attention_mask).logits
    loss = loss_fn(out, labels)
    accuracy_batch = compute_accuracy_batch(out, labels)

    return loss.item(), accuracy_batch, out
def eval_model_store_preds(model, dataloader, loss_fn, device, is_syntax_enhanced=False):
    """
    Evaluates a PyTorch model on a given DataLoader and stores the model's predictions.

    Args:
        model: A PyTorch model to evaluate.
        dataloader: A PyTorch DataLoader object containing the data to evaluate.
        loss_fn: A PyTorch loss function to use for evaluation.
        device: The device to use for evaluation (e.g., "cpu" or "cuda").
        is_syntax_enhanced: A boolean indicating whether to use the syntax-enhanced model (default False).

    Returns:
        A tuple containing the mean loss, mean accuracy, and all model predictions.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize losses, accuracies, and predictions
    losses, accuracies = [], []
    preds = []

    # Evaluate each batch in the DataLoader
    with torch.no_grad():
        for batch in dataloader:
            loss_batch, accuracy_batch, batch_preds = eval_batch_store_preds(model, batch, loss_fn, device, is_syntax_enhanced)
            preds.append(batch_preds)
            losses.append(loss_batch)
            accuracies.append(accuracy_batch)

    # Return mean loss, mean accuracy, and all predictions
    return np.mean(losses), np.mean(accuracies), preds
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
