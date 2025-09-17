import re
import os
import asyncio
import pandas as pd
from ask_llm import create_client, ask_gpt

def replace_non_alphanumeric_with_space(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
def remove_parentheses_component(text):
    cleaned_text = re.sub(r'\([^)]*\)', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()
def is_valid_non_letter_string(s):
    contains_digit = bool(re.search(r'\d', s))
    no_letters = not bool(re.search(r'[a-zA-Z]', s))
    return contains_digit and no_letters

def prompt(all_topics, example):
    input = f'The tables from the dataset are clustered into different groups. The topics of all groups are:\n{all_topics}\n'
    input += '\nNow, there are ambiguous tokens from each column name. They are ambiguous because LLM expands them into different expansions. For example:\n'
    input += example
    input += """\n\n\nBased on the context of the whole dataset, does this token appear to be a commonly used special token in this dataset or just a random abbreviation whose meaning may vary in different tables? If it is a common token, what is the most possible expansion of this token? Understand the whole dataset first and reason. Justify your answer. At the end, after '### Final Answer': output your answer for this ambiguous token in the format of token: answer, the answer is either the most possible expansion of this token or 'vary based on context"."""
    messages = [
        {'role': 'developer', 'content': 'You are a helpful assistant, answer the question from the user and reply in the required format.'},
        {'role': 'user', 'content': input}
    ]
    return messages

async def token_revision(df_explode, DATASET, TABLE_NAME, COLUMN_NAME, api_key):
    all_rules = df_explode[['rewrite_rules']].explode('rewrite_rules')
    all_rules = all_rules[all_rules['rewrite_rules'].apply(lambda x: isinstance(x, list) and len(x)==2)]
    all_rules['rewrite_rules'] = all_rules['rewrite_rules'].apply(lambda x: [i.lower().strip() for i in x])
    all_rules['token'] = all_rules['rewrite_rules'].apply(lambda x: x[0])
    all_rules['expansion'] = all_rules['rewrite_rules'].apply(lambda x: x[1])
    agg_tokens = all_rules.groupby(['token', 'expansion']).count().reset_index().rename(columns = {'rewrite_rules': 'count'})
    string_tokens = agg_tokens[agg_tokens['token'].apply(lambda x: not is_valid_non_letter_string(x))]
    rules = string_tokens[(string_tokens['expansion'].apply(lambda x: len(x) > 0)) & (string_tokens['token'] != "")]
    rules['count_exp'] = rules.apply(lambda x: [x['expansion'], x['count']], axis = 1)
    rules = rules.groupby('token').agg({'count_exp': list}).reset_index().set_index('token')['count_exp'].to_dict()
    rules = {k:v for k, v in rules.items() if (not k.isdigit()) and len(k) > 1 and len(v) > 1}
    print("number of rules", len(rules))

    all_topics = list(df_explode['topic_group'].unique())

    new = df_explode.groupby(TABLE_NAME).agg({
        COLUMN_NAME: list,
        'rewrite_rules': list
    }).reset_index()
    new['affected_tokens'] = new.apply(lambda x: set(sum([[m[0].lower() for m in i if m[0].lower() in rules] for i in x['rewrite_rules']], [])), axis = 1)
    verify = new[new['affected_tokens'] != set()]

    questions = []
    for token in rules:
        cur_q = ""
        if len(token) == 1:
            continue
        cur_q += f'ambiguous token: {token}\n'
        for exp in rules[token]:
            cur_q += f'possible expansion and count: {exp}\n'
            current = verify[verify['rewrite_rules'].apply(lambda x: [token, exp[0]] in  [[m.lower() for m in i] for i in sum(x, [])])]
            for _, s in current.sample(min(len(current), 1)).iterrows():
                cur_q += f'here is a sample table schema having this expansion: {s[TABLE_NAME]} {s[COLUMN_NAME]}\n'
        questions.append(cur_q)
    tmp = pd.DataFrame({'initial_info': questions})
    tmp['prompt'] = tmp.apply(lambda x: prompt(all_topics, x['initial_info']), axis = 1)

    client = create_client(api_key)
    prompt_list = tmp['prompt'].tolist()
    tasks = [ask_gpt(client, "gpt-4o", p, 1, 60) for p in prompt_list]
    results = await asyncio.gather(*tasks)

    while len([i for i in results if i.startswith('[TIME OUT]')]) > 0:
        print("[TIMEOUT] count:", len([i for i in results if i.startswith('[TIME OUT]')]))
        timeout_index = [i for i in range(len(results)) if results[i].startswith('[TIME OUT]')]
        timeout_prompts = [prompt_list[i] for i in timeout_index]

        timeout_results = await asyncio.gather(*[ask_gpt(client, "gpt-4o", p, 1, 120) for p in timeout_prompts])
        for idx, res in zip(timeout_index, timeout_results):
            results[idx] = res
    print("[TIMEOUT] count:", len([i for i in results if i.startswith('[TIME OUT]')]))
    tmp['result'] = results

    FOLDER = f"Results/{DATASET}"
    os.makedirs(FOLDER, exist_ok=True)
    tmp.to_pickle(f'{FOLDER}/token_ambiguity.pkl')
    special = tmp['result'].apply(lambda x: x.split('### Final Answer:\n')[-1])
    special = special[special.apply(lambda x: 'vary based on context' not in x)]
    s_t = {m[0]: m[1] for m in [i.split(': ') for i in list(special.values)]}

    df_explode['changed_rules'] = df_explode['rewrite_rules'].apply(lambda x: [[m.lower() for m in i] for i in x])
    df_explode['changed_rules'] = df_explode['changed_rules'].apply(lambda x: [[i[0], s_t[i[0]]] if i[0] in s_t else i for i in x ])
    df_explode['changed_rules'] = df_explode['changed_rules'].apply(lambda x: [[i[0], i[0]] if i[0].isdigit() else i for i in x])
    df_explode['changed_prediction'] = df_explode['changed_rules'].apply(lambda x: " ".join([i[-1] for i in x]))

    return df_explode[[TABLE_NAME, COLUMN_NAME, 'changed_prediction']].rename(columns = {'changed_prediction': 'PREDICTION'})


