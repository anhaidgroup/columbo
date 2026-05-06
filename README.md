<p align="center">
  <img src="logo.png" alt="Columbo Logo" width="200"/>
</p>

# Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2508.09403-b31b1b.svg)](https://arxiv.org/abs/2508.09403)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)

## Overview

This repository contains the official implementation for the EMNLP 2025 Findings paper: *Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models*.

Columbo is an LLM-based system that expands abbreviated column names in tabular data into their full, human-readable forms — for example, turning `e_sal` into `employee salary`. Abbreviated column names are pervasive in real-world databases and data lakes, making it difficult for both humans and downstream systems (e.g., data integration, question answering, schema matching) to interpret table schemas. Columbo addresses this problem by exploiting table context, rules, in-context demos, chain-of-thought reasoning, and token-level analysis to produce accurate, interpretable expansions.

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


# Citation
If you use Columbo in your work, please cite our EMNLP 2025 paper:

```bibtex
@inproceedings{cai2025columbo,
  title={Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models},
  author={Cai, Ting and Sheen, Stephen and Doan, AnHai},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}
```
