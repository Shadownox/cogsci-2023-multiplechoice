import pandas as pd
import numpy as np
import ccobra
import json

response_list = [x.lower() for x in ccobra.syllogistic.RESPONSES]

multi_df = pd.read_csv("../data/multiple_choice.csv")

def get_mat(df, weighted=True):
    mat = np.zeros((64,9))
    num_persons = len(np.unique(df["id"]))
    
    mfp = {}
    
    num_responses_without_nvc = []
    num_responses = []

    for _, row in df.iterrows():
        responses = row["enc_responses"]
        
        responses = eval(responses)
        num_responses.append(len(responses))
        if "nvc" not in responses:
            num_responses_without_nvc.append(len(responses))
        
        syl = row["enc_task"]
        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syl)
        
        if syl not in mfp:
            mfp[syl] = []
        mfp[syl].append(sorted(responses))
        
        for resp in responses:
            resp_idx = response_list.index(resp)
            score = 1
            if weighted:
                score = 1 / len(responses)
            mat[syl_idx, resp_idx] += score / num_persons
    
    mfp_res = {}
    for key, value in mfp.items():
        value = [tuple(x) for x in value]

        numbers = dict(zip(*np.unique(value, return_counts=True)))
        numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)

        mfp_res[key] = [numbers[0]]
        if numbers[0][1] == numbers[1][1]:
            mfp_res[key].append(numbers[1])
    
    print("Number of responses: avg={}, SD={}".format(np.mean(num_responses), np.std(num_responses)))
    print("Number of responses without NVC: avg={}, SD={}".format(np.mean(num_responses_without_nvc), np.std(num_responses_without_nvc)))
    return mat, mfp_res

mat, mfp_dict = get_mat(multi_df)
mfp_storage = {}

for syl, mfps in mfp_dict.items():
    syl_patterns = []
    for mfp in mfps:
        syl_patterns.append([x for x in mfp[0]])
    mfp_storage[syl] = syl_patterns

with open('../model_eval/individual/mfa/mfa.json', 'w', encoding ='utf8') as json_file:
    json.dump(mfp_storage, json_file, ensure_ascii = False)
    