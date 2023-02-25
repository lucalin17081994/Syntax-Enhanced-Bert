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
    return ((np.asarray(preds) == np.asarray(true)).mean() * 100)

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
