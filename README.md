# Syntax-Enhanced-Bert
Utrecht University Masters Thesis year 2022/2023

Studying the effect of enhancing Bert-based models with syntax on the task of Natural Language Inference

# Diagrams:
[Data Flow View](Functional_view.drawio.pdf)

# Notes:
- Organization into subfolders happened after I was done with the thesis. Some paths used in the .py or .ipynb files may need to be changed.
- For logging to wandb, replace '...' with your own API key.
- Snellius/HPC jobs are the .sh files. In total, there are three main .py files: SNLI dataset (Main.py), MNLI+other datasets (MNLI_train.py), and using glove embeddings (Glove.py) with a simpler bi-lstm model. SNLI and MNLI can probably be merged since they use the same model.

# Abstract

This Master's Thesis presents an exploration of incorporating syntax trees into pre-trained Large Language Models (LLMs) for the task of Natural Language Inference (NLI). NLI is an important task for evaluating language models' ability to predict the entailment relationship between two sentences, thus showcasing a model's capacity for Natural Language Understanding (NLU). This study predominantly focuses on the BERT-base-uncased model, assessing the effects of enhancing it with an inductive bias toward linguistically derived syntactic trees using Graph Convolutional Networks, and the effects on performance on various NLI benchmark datasets and out-of-domain evaluation sets. While earlier research has delved into the impacts of enhancing LLMs with dependency structures, the effects of incorporating constituency structures and combining both parsing techniques remain largely unexplored. Experimental results reveal that while enhancement of BERT with syntactic structures does not notably benefit generic large-scale NLI datasets, it significantly aids models in scenarios where the underlying syntactic structure is important for the inference task, such as in semi-automatically generated datasets. This is particularly evident when training data is scarce, a common challenge in many real-world applications. Results further show that of the two investigated syntactic structures, constituency structures provide the most benefits in learning representations for monotonicity reasoning, an important skill that requires the ability to capture interactions between lexical and syntactic structures. Furthermore, we demonstrate that constituency parsing can help the BERT model learn useful representations for the syntactic structure of passive sentences, an area identified in previous research as a shortcoming of BERT. 

