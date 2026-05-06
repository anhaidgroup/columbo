<p align="center">
  <img src="logo.png" alt="Columbo Logo" width="200"/>
</p>

# Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2508.09403-b31b1b.svg)](https://arxiv.org/abs/2508.09403)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)

## Overview

Columbo is an LLM-based system that expands abbreviated column names in tabular data into their full, human-readable forms — for example, turning `e_sal` into `employee salary`. Abbreviated column names are pervasive in real-world databases and data lakes, making it difficult for both humans and downstream systems (e.g., data integration, question answering, schema matching) to interpret table schemas. Columbo addresses this problem by exploiting table context, rules, in-context demos, chain-of-thought reasoning, and token-level analysis to produce accurate, interpretable expansions.

# Installation

This codebase was developed and tested with **Python 3.10.12**. To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Then, download the spaCy English model, which is used for lemmatization (e.g., normalizing "salaries" → "salary") when computing the evaluation metrics:

```bash
python -m spacy download en_core_web_sm
```

# Dataset

Three benchmark datasets are provided in `./clean_data/` for evaluating column name expansion:

| Dataset | # Columns | Table column | Column name column | Gold label column |
|---|---|---|---|---|
| `AdventureWork` | 825 | `Table` | `COLUMN_NAME_1` | `GT_LABEL_1` |
| `EDI_demo` | 3,830 | `table_name` | `column_name` | `gt_label` |
| `nameguess` | 9,196 | — | `technical_name` | `gt_label` |

Each dataset folder contains:
- `gold.pkl` — a pandas DataFrame with table names, abbreviated column names, and their gold expansions
- `synonyms.json` — synonyms for gold expansion words, used during evaluation
- `stopwords.json` — stopwords to ignore during evaluation

# Usage

To run Columbo on one of the provided datasets, pass the dataset folder name and the corresponding column name arguments. For example, to run on the **AdventureWork** dataset:

```bash
python main.py \
  --dataset AdventureWork \
  --table Table \
  --column COLUMN_NAME_1 \
  --gt_label GT_LABEL_1 \
  --api_key [YOUR_OPENAI_API_KEY]
```

**Using your own data**

If you have a CSV file with your own tables and column names, you can run Columbo on it as follows:

1. Create a folder for your dataset under `clean_data/`:
```
clean_data/<YOUR_DATASET>/
```

2. Convert your CSV to a pickle file and save it as `gold.pkl`. Your CSV must have at least three columns: one for the table name, one for the abbreviated column name, and one for the ground truth expansion (used for evaluation):
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
df.to_pickle('clean_data/<YOUR_DATASET>/gold.pkl')
```

3. Create empty `synonyms.json` and `stopwords.json` files:
```bash
echo '{}' > clean_data/<YOUR_DATASET>/synonyms.json
echo '{}' > clean_data/<YOUR_DATASET>/stopwords.json
```

4. Run Columbo with the column names from your CSV:
```bash
python main.py \
  --dataset <YOUR_DATASET> \
  --table <TABLE_COLUMN> \
  --column <COLUMN_NAME_COLUMN> \
  --gt_label <GOLD_LABEL_COLUMN> \
  --api_key [YOUR_OPENAI_API_KEY]
```


# Citation
If you use Columbo in your work, please cite our EMNLP 2025 Findings paper:

```bibtex
@inproceedings{cai2025columbo,
  title={Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models},
  author={Cai, Ting and Sheen, Stephen and Doan, AnHai},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
  year={2025}
}
```
