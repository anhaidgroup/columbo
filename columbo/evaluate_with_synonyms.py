import re
import spacy
import pandas as pd
import os
from src.metric import BNGMetrics
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers = os.cpu_count())

def group_by_sizes(elements, sizes):
    grouped = []
    index = 0  # Keep track of the current index in elements
    
    for size in sizes:
        grouped.append(elements[index:index + size])
        index += size  # Move index forward by the current group size
    
    return grouped


def split_text_numbers(text):
    text = re.sub(r"[_/\-(),.\\]", " ", text)
    text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def refine_LLM_expansion(stopwords, expansion):
    return " ".join([i for i in split_text_numbers(expansion).split(" ") if i.lower() not in stopwords])


def generate_synonym_sentences_dp(input_string, synonym_table, stopwords):
    tokens = input_string.split()  # Tokenize input
    max_phrase_length = max((len(phrase.split()) for phrase in synonym_table), default=1)
    n = len(tokens)
    dp = [[] for _ in range(n + 1)]
    dp[n] = [""]

    contains_synonym = False  

    for i in range(n - 1, -1, -1):
        phrase_matched = False

        for phrase_length in range(1, max_phrase_length + 1):
            if i + phrase_length <= n:
                phrase = " ".join(tokens[i:i + phrase_length])
                # print(f"Checking phrase: '{phrase}'")  # Debugging output

                if phrase in synonym_table:
                    contains_synonym = True
                    phrase_matched = True
                    synonyms = synonym_table[phrase] + [phrase]
                    
                    for synonym in synonyms:
                        for sentence in dp[i + phrase_length]:
                            new_sentence = synonym + (" " + sentence if sentence else "")
                            dp[i].append(new_sentence)

        if not phrase_matched:
            for sentence in dp[i + 1]:
                dp[i].append(tokens[i] + (" " + sentence if sentence else ""))

    if not contains_synonym:
        # return [" ".join(tokens)]
        output = [" ".join(tokens)]
    else:
        output = dp[0]

    # output = dp[0]
    for i in range(len(output)):
        output[i] = " ".join([elm for elm in split_text_numbers(output[i]).split(" ") if elm not in stopwords])
    
    # return dp[0]
    return output

def evaluate_w_synonyms(pred_df, gold, synonyms, stopwords, TABLE_NAME, COLUMN_NAME, GT_LABEL, PRED):
    new_synonyms = {}
    for i in synonyms:
        if isinstance(synonyms[i], str):
            new_synonyms[i.lower()] = set([synonyms[i].lower()])
            new_synonyms[synonyms[i].lower()] = set([i.lower()])
        else:
            new_synonyms[i.lower()] = set([m.lower() for m in synonyms[i]])
            for val in synonyms[i]:
                new_synonyms[val.lower()] = set([i.lower()] + [x.lower() for x in synonyms[i] if x != val])
    synonyms_table = {k:list(v) for k, v in new_synonyms.items()}

    df_explode = gold.merge(pred_df, on=[TABLE_NAME, COLUMN_NAME], how='left')

    calculate = df_explode[(df_explode[GT_LABEL].apply(lambda x: (not pd.isna(x)) and x != "")) & (df_explode[PRED].apply(lambda x: (not pd.isna(x)) and x != ""))]
    calculate['gt_label_refined'] = calculate[GT_LABEL].apply(lambda x:generate_synonym_sentences_dp(x.lower(), synonyms_table, stopwords))

    calculate['LLM_expansion_refined'] = calculate.apply(lambda x: [refine_LLM_expansion(stopwords, x[PRED])]*len(x['gt_label_refined']), axis = 1)


    metric_generator = BNGMetrics(['squad'], device='cpu')

    results = calculate.parallel_apply(lambda x: metric_generator.compute_scores(predictions = x['LLM_expansion_refined'], references = x['gt_label_refined'], level = 'individual'), axis = 1)

    print("exact match score with synonyms is:", results.apply(lambda x: max(x['individual_squad-em'])).sum() / len(gold[~gold[GT_LABEL].isna()]))
    print('f1 score with synonyms is:', results.apply(lambda x: max(x['individual_squad-f1'])).sum() / len(gold[~gold[GT_LABEL].isna()]) )

    # metric_generator = BNGMetrics(['bertscore-f1'], device='cuda')
    # results = metric_generator.compute_scores(predictions=sum(calculate['LLM_expansion_refined'], []), references = sum(calculate['gt_label_refined'], []), level = "individual")
    # sizes = calculate['gt_label_refined'].apply(lambda x: len(x)).values

    # print('bert f1 with synonyms is', sum([max(i) for i in group_by_sizes(results['individual_bertscore-f1'], sizes)]) / len(gold[~gold[GT_LABEL].isna()]))

