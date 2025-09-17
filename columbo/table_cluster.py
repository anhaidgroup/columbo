import pandas as pd
import numpy as np
import json
import re
import os
from ask_llm import create_client, ask_gpt
import asyncio
pd.options.mode.chained_assignment = None

def check_group(tn, groups):
    for key in groups.keys():
        if tn in groups[key]:
            return key
    return np.nan

async def table_clustering(gold, DATASET, TABLE_NAME, COLUMN_NAME, BATCH_SIZE, api_key, MODEL_NAME='gpt-4o', Temperature=0):

    client = create_client(api_key)

    result = []
    cur_df = gold.groupby(TABLE_NAME).agg({COLUMN_NAME: list}).reset_index()

    for i in range(0, len(cur_df) // BATCH_SIZE + 1):
        schema = ""
        for _, row in cur_df[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE)].iterrows():
            schema += row[TABLE_NAME] + " " + str(row[COLUMN_NAME])+'\n'
        cur_prompt = f"""
    Given the following abbreviated table schemas from the same dataset:
    {schema}
    Your job is to:

    1. Determine the overall context and domain of the dataset based on column names and structures.
    2. Explain how you identified the context and grouped the tables together. Group them into categories based on their function or thematic role.
    3. For each table, write a clear and concise topic that summarizes the purpose of the table (not a direct name expansion).
    a. These topics should help explain what kind of data is in each table and how it's used.
    4. At the end, output the grouped table topics, a JSON object where:
    i. Each key is a group topic
    ii. Each value is a dictionary where keys are table names and values are table topics.
    """
        messages = [
            {'role': 'developer', 'content': 'You are a helpful assistant, answer the question from the user and reply in the required format.'},
            {'role': 'user', 'content': cur_prompt}
        ]
        result.append(messages)
    # print(cur_prompt)

    result_df = pd.DataFrame({'prompts': result})
    prompt_list = result_df['prompts'].tolist()
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
    result_df['results'] = results


    FOLDER = f'Results/{DATASET}'
    os.makedirs(FOLDER, exist_ok=True)

    with open(f'{FOLDER}/cluster_topics.json', 'w') as file:
        topics = [json.loads(i.split('```json')[-1].split('```')[0]) for i in result_df['results']]
        json.dump(topics, file, indent=4)
    print(f"cluster topics saved to {FOLDER}/cluster_topics.json")

    group_topics = {}
    for cur in topics:
        for g_topic in cur:
            if g_topic not in group_topics:
                group_topics[g_topic] = cur[g_topic]
            else:
                group_topics[g_topic] = group_topics[g_topic] | cur[g_topic]
    print("number of clusters:", len(group_topics))
    all_topics = {}
    for key in group_topics:
        all_topics = all_topics | group_topics[key]

    gold['topics'] = gold[TABLE_NAME].apply(lambda x: all_topics[x] if x in all_topics else np.nan)

    gold['topic_group'] = gold[TABLE_NAME].apply(lambda x: check_group(x, group_topics))

    return gold
