import os
import re
import asyncio
from ask_llm import create_client, ask_gpt

def is_valid_non_letter_string(s):
    contains_digit = bool(re.search(r'\d', s))
    no_letters = not bool(re.search(r'[a-zA-Z]', s))
    return contains_digit and no_letters

def group_every_K(lst, K):
    return [lst[i:i+K] for i in range(0, len(lst), K)]
def add_topic(row, df, topic_group, TABLE_NAME, COLUMN_NAME):
    system_prompt = "You are a helpful assistant, answer the question from the user and reply in the same format."
    
    new_query = f"Given a list of tables and their topics in the same dataset:\n{df[df['topic_group']==topic_group][[TABLE_NAME, 'topics']].drop_duplicates().values.tolist()}\n\n"
    t_name = row[TABLE_NAME]
    c_names = " | ".join(row[COLUMN_NAME])
    query_2 = f"""
Your task is to expand abbreviated column names into full-form phrases. 
You should return the tokens of each column name and their associated expansions.
First reason step by step and then return your final answer. 
Follow the guidelines below when you expand:
1. Expand all abbreviations in the column names. 
2. Expand chemical symbols and units of measure to their full names.
3. Do not expand or mutate numbers. 
4. Do not add extra words or explanations.
5. Maintain the original order of tokens in the expansion.
6. The tokens should be as concise and simple as possible.
7. Only provide 1 expansion for each token, even if you are uncertain, output the most possible one.
Do not put ambiguous or need context after your expansion.
8. If the token is not abbreviated, the expansion should be itself, do not paraphrase or add other words around it.


[Question]
As abbreviations of column names from a table named Prchs_info, c_name | pCd | dt stand for? Think step by step
[Answer]
### Reasoning 
1. **Identify the table name context:**  
   - The table is named **"Prchs_info"**, which likely stands for **"Purchase Information"**.  
   - This suggests that the column names are related to purchase details.  

2. **Break down each column name into tokens and infer meanings:**  
   - **c_name** → `c`, `name`  
     - `c` is commonly used for **Customer** in business contexts.  
     - `name` clearly refers to **Name**.  
     - Together, `c_name` most likely means **Customer Name**.  

   - **pCd** → `p`, `Cd`  
     - `p` is frequently used for **Product** in sales or inventory tables.  
     - `Cd` is a common abbreviation for **Code**.  
     - Together, `pCd` most likely means **Product Code**.  

   - **dt** → `dt`  
     - `dt` is a standard abbreviation for **Date**.  
     - In the context of a purchase table, it likely refers to **Purchase Date**.  

### Final Answer  
c_name: c → Customer, name → Name
pCd: p → Product, Cd → Code
dt: dt → Date


[Question]
As abbreviations of column names from a table named Emp_info, emp_id | emp_nm | dept_cd | doj | sal_amt | mgr_id stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"Emp_info"**, which likely stands for **"Employee Information"**.  
   - This suggests that the column names are related to employee records.  

2. **Break down each column name into tokens and infer meanings:**  
   - **emp_id** → `emp`, `id`  
     - `emp` is commonly used for **Employee**.  
     - `id` stands for **Identifier**.  
     - Together, `emp_id` means **Employee Identifier**.  

   - **emp_nm** → `emp`, `nm`  
     - `emp` stands for **Employee**.  
     - `nm` is a common abbreviation for **Name**.  
     - Together, `emp_nm` means **Employee Name**.  

   - **dept_cd** → `dept`, `cd`  
     - `dept` is short for **Department**.  
     - `cd` stands for **Code**.  
     - Together, `dept_cd` means **Department Code**.  

   - **doj** → `doj`  
     - `doj` is commonly used for **Date of Joining**.  

   - **sal_amt** → `sal`, `amt`  
     - `sal` stands for **Salary**.  
     - `amt` stands for **Amount**.  
     - Together, `sal_amt` means **Salary Amount**.  

   - **mgr_id** → `mgr`, `id`  
     - `mgr` is short for **Manager**.  
     - `id` stands for **Identifier**.  
     - Together, `mgr_id` means **Manager Identifier**.  

### Final Answer
emp_id: emp → Employee, id → Identifier
emp_nm: emp → Employee, nm → Name
dept_cd: dept → Department, cd → Code
doj: doj → Date of Joining
sal_amt: sal → Salary, amt → Amount
mgr_id: mgr → Manager, id → Identifier

[Question]
As abbreviations of column names from a table named AirQ_data, stn_id | pm25_lvl | pm10_lvl | no2_lvl | co_lvl | temp_C | rec_dt stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"AirQ_data"**, which likely stands for **"Air Quality Data"**.  
   - This suggests that the column names are related to air quality measurements.  

2. **Break down each column name into tokens and infer meanings:**  
   - **stn_id** → `stn`, `id`  
     - `stn` stands for **Station**.  
     - `id` stands for **Identifier**.  
     - Together, `stn_id` means **Station Identifier**.  

   - **pm25_lvl** → `pm25`, `lvl`  
     - `pm25` stands for **Particulate Matter 2.5**.  
     - `lvl` stands for **Level**.  
     - Together, `pm25_lvl` means **Particulate Matter 2.5 Level**.  

   - **pm10_lvl** → `pm10`, `lvl`  
     - `pm10` stands for **Particulate Matter 10**.  
     - `lvl` stands for **Level**.  
     - Together, `pm10_lvl` means **Particulate Matter 10 Level**.  

   - **no2_lvl** → `no2`, `lvl`  
     - `no2` stands for **Nitrogen Dioxide**.  
     - `lvl` stands for **Level**.  
     - Together, `no2_lvl` means **Nitrogen Dioxide Level**.  

   - **co_lvl** → `co`, `lvl`  
     - `co` stands for **Carbon Monoxide**.  
     - `lvl` stands for **Level**.  
     - Together, `co_lvl` means **Carbon Monoxide Level**.  

   - **temp_C** → `temp`, `C`  
     - `temp` stands for **Temperature**.  
     - `C` stands for **Celsius**.  
     - Together, `temp_C` means **Temperature Celsius**.  

   - **rec_dt** → `rec`, `dt`  
     - `rec` stands for **Recorded**.  
     - `dt` stands for **Date**.  
     - Together, `rec_dt` means **Recorded Date**.  

### Final Answer  
stn_id: stn → Station, id → Identifier
pm25_lvl: pm25 → Particulate Matter 2.5, lvl → Level
pm10_lvl: pm10 → Particulate Matter 10, lvl →Level
no2_lvl: no2 → Nitrogen Dioxide, lvl → Level
co_lvl: co → Carbon Monoxide, lvl → Level
temp_C: temp → Temperature, C → Celsius
rec_dt: rec → Recorded, dt → Date  


[Question]
As abbreviations of column names from a table named Pop_census, hh_id | res_cty | age_grp | gender_cd | edu_lvl | occ_cd stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"Pop_census"**, which likely stands for **"Population Census"**.  
   - This suggests that the column names are related to demographic and census data.  

2. **Break down each column name into tokens and infer meanings:**  
   - **hh_id** → `hh`, `id`  
     - `hh` stands for **Household**.  
     - `id` stands for **Identifier**.  
     - Together, `hh_id` means **Household Identifier**.  

   - **res_cty** → `res`, `cty`  
     - `res` stands for **Residential**.  
     - `cty` stands for **County**.  
     - Together, `res_cty` means **Residential County**.  

   - **age_grp** → `age`, `grp`  
     - `age` stands for **Age**.  
     - `grp` stands for **Group**.  
     - Together, `age_grp` means **Age Group**.  

   - **gender_cd** → `gender`, `cd`  
     - `gender` stands for **Gender**.  
     - `cd` stands for **Code**.  
     - Together, `gender_cd` means **Gender Code**.  

   - **edu_lvl** → `edu`, `lvl`  
     - `edu` stands for **Education**.  
     - `lvl` stands for **Level**.  
     - Together, `edu_lvl` means **Education Level**.  

   - **occ_cd** → `occ`, `cd`  
     - `occ` stands for **Occupation**.  
     - `cd` stands for **Code**.  
     - Together, `occ_cd` means **Occupation Code**.  

### Final Answer  
hh_id: hh → Household, id → Identifier
res_cty: res → Residential, cty → County
age_grp: age → Age, grp → Group
gender_cd: gender → Gender, cd → Code
edu_lvl: edu → Education, lvl → Level
occ_cd: occ → Occupation, cd → Code  

[Question]
As abbreviations of column names from a table named {t_name}, {c_names} stand for
[Answer]
"""

    messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": new_query+query_2}
    ]
    return messages


async def column_expansion(result_df, K, DATASET, TABLE_NAME, COLUMN_NAME, api_key):
    topics_table = result_df.groupby(TABLE_NAME).agg({
        'topics': 'first',
        'topic_group': 'first'
    }).reset_index()

    new = result_df.groupby(TABLE_NAME).agg({
        COLUMN_NAME: list
    }).reset_index()
    new[COLUMN_NAME] = new[COLUMN_NAME].apply(lambda x: group_every_K(x, K))
    new_explode = new.explode(COLUMN_NAME).reset_index(drop=True)
    new_explode_topics = new_explode.merge(topics_table, on = TABLE_NAME, how = 'left')
    # new_explode_topics[COLUMN_NAME].apply(lambda x: len(x))

    df = new_explode_topics.copy()
    df['cluster_prompt'] = df.apply(lambda x: add_topic(x, df, x['topic_group'], TABLE_NAME, COLUMN_NAME), axis = 1)
    # for i in df['cluster_prompt'].iloc[0]:
    #     print(i['content'])

    print("start expanding...")

    client = create_client(api_key)
    prompt_list = df['cluster_prompt'].tolist()
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
    df['cluster_results'] = results

    FOLDER = f'Results/{DATASET}'
    os.makedirs(FOLDER, exist_ok=True)
    df.to_pickle(f'{FOLDER}/raw_expansion_output.pkl')
    
    print(f"{len(df)} queries asked.\nRaw expansion output saved! Now start the column expansion extraction step...")

    answer = 'cluster_results'
    df_filter = df[~df[answer].isna()]
    df_filter['expansion'] = df_filter[answer].apply(lambda x: [i for i in x.split('### Final Answer')[1].split('\n') if i.strip() != ""])
    df_filter = df_filter[df_filter.apply(lambda x: len(x[COLUMN_NAME]) == len(x['expansion']), axis = 1)]
    print(round(len(df_filter) / len(df), 4)*100, "% passed the filtering step.")

    df_explode = df_filter.explode([COLUMN_NAME, 'expansion'])

    pattern = r"([\w/^%():#/\.\-\s]+?)\s*→\s*([^→]+?)(?=(?:,\s*[\w/^%():#/\.\-\s]+?\s*→|$))"
    df_explode['rewrite_rules'] = df_explode['expansion'].apply(lambda x: [[key.strip(), value.strip()] for key, value in re.findall(pattern, x.split(": ")[-1])])

    df_explode['rewrite_rules'] = df_explode['rewrite_rules'].apply(lambda x: [[i[0], i[0]] if is_valid_non_letter_string(i[0]) else i for i in x])
    df_explode['prediction'] = df_explode['rewrite_rules'].apply(lambda x: " ".join([i[-1] for i in x]))

    merged_df = result_df.merge(df_explode[[TABLE_NAME, COLUMN_NAME, 'rewrite_rules', 'prediction']], on = [TABLE_NAME, COLUMN_NAME], how = 'left')
    return merged_df
