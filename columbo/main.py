import pandas as pd
import asyncio
import argparse
import json
from table_cluster import table_clustering
from column_expand import column_expansion
from token_revise import token_revision
from evaluate_with_synonyms import evaluate_w_synonyms

async def main(args):
    DATASET = args.dataset
    TABLE_NAME = args.table
    COLUMN_NAME = args.column
    GT_LABEL = args.gt_label
    MODEL_NAME = args.model
    Temperature = args.temperature
    BATCH_SIZE = args.batch_size
    api_key = args.api_key

    gold = pd.read_pickle(f'../clean_data/{DATASET}/gold.pkl')
    print("Gold data loaded. Starting table clustering...")
    result_df = await table_clustering(gold[[TABLE_NAME, COLUMN_NAME]], DATASET, TABLE_NAME, COLUMN_NAME, BATCH_SIZE, api_key, MODEL_NAME=MODEL_NAME, Temperature=Temperature)
    result_df.to_pickle('result_df.pkl')

    K = args.k
    # result_df = pd.read_pickle('result_df.pkl')
    exp_df = await column_expansion(result_df, K, DATASET, TABLE_NAME, COLUMN_NAME, api_key)
    exp_df.to_pickle('exp_df.pkl')

    # exp_df = pd.read_pickle('exp_df.pkl')
    pred_df = await token_revision(exp_df, DATASET, TABLE_NAME, COLUMN_NAME, api_key)
    pred_df.to_pickle('pred_df.pkl')

    # pred_df = pd.read_pickle('pred_df.pkl')
    with open(f'../clean_data/{DATASET}/synonyms.json', 'r') as file:
        synonyms = json.load(file)
    with open(f'../clean_data/{DATASET}/stopwords.json', 'r') as file:
        stopwords = json.load(file)

    # gold = pd.read_pickle(f'../clean_data/{DATASET}/gold.pkl')

    PRED = 'PREDICTION'
    evaluate_w_synonyms(pred_df, gold, synonyms, stopwords, TABLE_NAME, COLUMN_NAME, GT_LABEL, PRED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--table', type=str, required=True, help='Table name')
    parser.add_argument('--column', type=str, required=True, help='Column name')
    parser.add_argument('--gt_label', type=str, required=True, help='Ground truth label column name')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for model')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for processing')
    parser.add_argument('--k', type=int, default=10, help='K value for column expansion')
    parser.add_argument('--api_key', type=str, required=True, help='API key for access')

    args = parser.parse_args()
    asyncio.run(main(args))