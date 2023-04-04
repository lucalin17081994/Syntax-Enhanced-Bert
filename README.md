# Syntax-Enhanced-Bert
Utrecht University Masters Thesis year 2022/2023

Studying the effect of enhancing Bert-based models with syntax on the task of Natural Language Inference

Notebook Links:

- <a href="https://colab.research.google.com/drive/1YLzZESUyOR1mSlqUYb1zoz4uHfMMWfCA?usp=sharing" target="_blank">Parsing datasets using stanza, from textfile</a>
- <a href="https://colab.research.google.com/drive/1ojdWbJXgqBNc2vdzo-3fDR_2XyididsR?usp=sharing" target="_blank">Parse HANS dataset using stanza, from huggingface datasets</a>
- <a href="https://colab.research.google.com/drive/1yGxFyEvTUY_ucoIZdyhfVIOdiJcgY49t?usp=sharing" target="_blank">Parse MED dataset using stanza, from textfile</a>
- <a href="https://colab.research.google.com/drive/1xPgVxDzchv7ZBsKzJ4VoAdnp23q-tkNu#scrollTo=yNak14_2ke5Y" target="_blank">training notebook</a>
- <a href="https://colab.research.google.com/drive/1920fAqJ-niy9F-w9AZeoCFfbbWxwX4D8#scrollTo=AK6wfzlM6aRD" target="_blank">evaluation notebook</a>

Pandas df should have the following columns after parsing the data:
1. gold label
2. sentence1 : full sentence as string
3. sentence2 : full sentence as string
4. text_sentence1 : list containing words of sentence1
5. text_sentence2 : list containing words of sentence2
6. pos_sentence1 : list of pos tags
7. pos_sentence2 : list of pos tags
8. heads_sentence1 : list of integers
9: heads_sentence2 : list of integers
10. deprel_sentence1 : list of strings
11. deprel_sentence2 : list of strings
12. sentence1_parse : constituency parse as a string
13. sentence2_parse : constituency parse as a string

Note: I started this project with all above features and found out later that some were unnecessary. Due to time limitations I decided to simply drop the columns during runtime instead of dropping them in the file itself.


