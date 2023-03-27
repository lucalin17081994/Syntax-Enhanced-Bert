import math
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def tree_kernel_from_NLTK_string(tree1, tree2):
    """
    Compute the similarity between two constituency trees using the subtree kernel.

    The subtree kernel is a similarity measure that counts the number of common subtrees
    between two trees. It is defined as the size of the intersection of the set of
    subtrees of the first tree and the set of subtrees of the second tree.

    Parameters:
    -----------
    tree1 : nltk.Tree
        The first constituency tree.
    tree2 : nltk.Tree
        The second constituency tree.

    Returns:
    --------
    int
        The number of common subtrees between the two trees.
    """
    def subtrees(tree):
        return set([str(subtree) for subtree in tree.subtrees()])
    
    tree1_subtrees = subtrees(tree1)
    tree2_subtrees = subtrees(tree2)
    
    common_subtrees = tree1_subtrees.intersection(tree2_subtrees)
    # NTK = normalized tree kernel

    similarity = len(common_subtrees)
    NTK = similarity / math.sqrt(len(tree1_subtrees.intersection(tree1_subtrees)) * len(tree2_subtrees.intersection(tree2_subtrees)))
    return NTK

def calculate_const_tree_similarity(df):
    '''
    apply tree_kernel function to entire dataframe, first replace words with a dummy token
    '''

    def replace_words_not_in_dict(tree_str, c_dict):
      

        words_to_replace =re.findall(r"[\w']+|\(|\)", tree_str)
        # replace each word with 'x'
        for word in words_to_replace:
            if word not in c_dict:
                tree_str = tree_str.replace(word, 'x')
        return tree_str

    c_dict= {'(':0, ')':0, 'ROOT':0,'NP': 0, 'DT': 1, 'NN': 2, 'PP': 3, 'IN': 4, 'S': 5, 'VP': 6, 'VBZ': 7, 'ADJP': 8, 'VBN': 9, 'RP': 10, '.': 11, 'VBG': 12, 'PRP$': 13, ',': 14, 'ADVP': 15, 'RB': 16, 'NNS': 17, 'CC': 18, 'PRP': 19, 'VBP': 20, 'EX': 21, 'JJ': 22, 'JJR': 23, 'SBAR': 24, 'TO': 25, 'VB': 26, 'CD': 27, 'HYPH': 28, 'WHNP': 29, 'WP': 30, 'VBD': 31, 'PRT': 32, 'NML': 33, 'NNP': 34, 'WHADVP': 35, 'WRB': 36, 'WDT': 37, 'POS': 38, 'QP': 39, 'WHADJP': 40, 'MD': 41, '``': 42, 'NNPS': 43, "''": 44, 'PDT': 45, 'RBR': 46, 'JJS': 47, 'UH': 48, 'UCP': 49, 'WHPP': 50, 'AFX': 51, 'SYM': 52, 'SINV': 53, 'X': 54, 'CONJP': 55, 'NFP': 56, 'FRAG': 57, 'WP$': 58, ':': 59, '-LRB-': 60, '-RRB-': 61, 'SBARQ': 62, 'SQ': 63, 'FW': 64, 'PRN': 65, 'INTJ': 66, 'RBS': 67, '$': 68, 'ADD': 69, 'RRC': 70, 'LST': 71, 'LS': 72, 'GW': 73}

    df['sentence1_parse'] = df['sentence1_parse'].apply(lambda x: replace_words_not_in_dict(x, c_dict))
    df['sentence2_parse'] = df['sentence2_parse'].apply(lambda x: replace_words_not_in_dict(x, c_dict))

    similarities = []
    for index, row in df.iterrows():
        tree1_str = row['sentence1_parse']
        tree2_str = row['sentence2_parse']
        
        tree1 = nltk.Tree.fromstring(tree1_str)
        tree2 = nltk.Tree.fromstring(tree2_str)
        
        similarity = tree_kernel_from_NLTK_string(tree1, tree2)
        similarities.append(similarity)
    
    return similarities
    
def tree_kernel(node1, node2):
    if not node1 or not node2:
        return 0

    if node1['label'] == node2['label']:
        num_common_subtrees = 1
        for i in range(len(node1['children'])):
            for j in range(len(node2['children'])):
                num_common_subtrees *= (1 + tree_kernel(node1['children'][i], node2['children'][j]))
    else:
        num_common_subtrees = 0

    return num_common_subtrees

def normalized_tree_kernel(tree1, tree2):
    kernel12 = tree_kernel(tree1, tree2)
    kernel11 = tree_kernel(tree1, tree1)
    kernel22 = tree_kernel(tree2, tree2)

    return kernel12 / ((kernel11 * kernel22) ** 0.5)

def create_dependency_tree(heads, rels):
    '''
    create dependency tree. heads and rels are lists representing dependency heads and relations
    '''
    tree = {}
    for i, head in enumerate(heads):
        node = {'label': rels[i], 'children': []}
        tree[i + 1] = node

    tree[0] = {'label': 'root', 'children': []}

    for i, head in enumerate(heads):
        tree[head]['children'].append(tree[i + 1])

    return tree[0]
def calculate_dep_tree_similarity(row):
    '''
    use .apply() function in pandas
    '''
    tree1 = create_dependency_tree(row['heads_sentence1'], row['deprel_sentence1'])
    tree2 = create_dependency_tree(row['heads_sentence2'], row['deprel_sentence2'])
    return normalized_tree_kernel(tree1, tree2)
def plot_tree_similarity(data, labels, model_name, dataset):
    # Concatenate the data columns into a single DataFrame
    combined_data = pd.concat(data, axis=1)

    # Compute the frequency distributions of the data
    bins = np.linspace(combined_data.min().min(), combined_data.max().max(), 20)

    # Create a line plot of the frequency distributions
    for i, data_col in enumerate(data):
        freq, _ = np.histogram(data_col, bins=bins, density=True)
        plt.plot(bins[:-1], freq, label=labels[i])

    # Set the axis labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Tree similarity, ' + model_name + ', ' + dataset)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    # Print mean values for each data column
    for i, data_col in enumerate(data):
        print(f"{labels[i]} Mean Tree Distance: {data_col.mean():.2f}")
