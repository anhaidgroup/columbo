# Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models
Repository for the Columbo paper, contains the data, method, and evaluation script. Columbo is a LLM-based solution to expand abbreviate column names into its full-form, e.g. "e_sal" to "employee salary"



# Installation
Recommend Python Environment: Python 3.10.12
Required packages: requirements.txt

Install necessary dictionary:
``python -m spacy download en_core_web_sm``

# Dataset
There are 3 datasets for evaluating column name expansion in the ./clean_data folder: NameGuess, EDI, AdventureWork. 

For each dataset: 
- gold.pkl contains the table name, column names and gold expansion of column names. 
- synonyms.json contains the synonyms for the full-form words. 
- stopwords.json contains the stopwords appeared in the dataset.

# Usage
An example usage to run Columbo on "AdventureWork" dataset is:

``python main.py   --dataset AdventureWork_1   --table Table   --column COLUMN_NAME_1   --gt_label GT_LABEL_1   --api_key [YOUR_OPENAI_API_KEY]``


# Cite
Please cite the paper if you use the codebase in your work

```bibtex
@article{cai2025columbo,
  title={Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models},
  author={Cai, Ting and Sheen, Stephen and Doan, AnHai},
  journal={arXiv preprint arXiv:2508.09403},
  year={2025}
}
```
