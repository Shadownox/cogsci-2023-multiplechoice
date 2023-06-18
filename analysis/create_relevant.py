import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import seaborn as sns
import ccobra
import json

response_list = [x.lower() for x in ccobra.syllogistic.RESPONSES]

khemlani_df = pd.read_csv("../data/khemlani2012.csv", sep=";")
khemlani_mat = np.zeros((64,9))
for syl_idx, syllog in enumerate(ccobra.syllogistic.SYLLOGISMS):
    row = khemlani_df[khemlani_df["Syllog"] == syllog]
    for resp_idx, resp in enumerate(ccobra.syllogistic.RESPONSES):
        val = row[resp].values[0]
        if np.isnan(val):
            val = 0

        khemlani_mat[syl_idx, resp_idx] = float(val)
khemlani_mat /= np.sum(khemlani_mat, axis=1, keepdims=True)

def gen_task_enc(elem):
    item = ccobra.Item(0, "syllogistic", elem, "single-choice", "", 0)
    syl = ccobra.syllogistic.Syllogism(item)
    return syl.encoded_task

def gen_resp_enc(elem):
    item = ccobra.Item(0, "syllogistic", elem["task"], "single-choice", "", 0)
    syl = ccobra.syllogistic.Syllogism(item)
    return "['{}']".format(syl.encode_response(elem["response"].split(";")).lower())

orig_df = pd.read_csv("../data/Ragni2016.csv")
orig_df["enc_task"] = orig_df["task"].apply(gen_task_enc)
orig_df["enc_responses"] = orig_df[["task", "response"]].apply(gen_resp_enc, axis=1)

multi_df = pd.read_csv("../data/multiple_choice.csv")

def get_mat(df, weighted=True):
    mat = np.zeros((64,9))
    num_persons = len(np.unique(df["id"]))

    for _, row in df.iterrows():
        responses = row["enc_responses"]
        
        responses = eval(responses)
        syl = row["enc_task"]
        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syl)
        
        for resp in responses:
            resp_idx = response_list.index(resp)
            score = 1
            if weighted:
                score = 1 / len(responses)
            mat[syl_idx, resp_idx] += score / num_persons
    return mat

def get_relevant_responses(df, mat=None, weighted=True):
    if mat is None:
        mat = get_mat(df, weighted=weighted)
    results = {}

    for y in range(64):
        responses = []
        for x in range(9):
            if mat[y, x] >= 0.16:
                responses.append(ccobra.syllogistic.RESPONSES[x])
        results[ccobra.syllogistic.SYLLOGISMS[y]] = responses
                
    return results

# Generate relevant-only matrices
responses_khem = get_relevant_responses(None, mat=khemlani_mat)
responses_ragni = get_relevant_responses(orig_df)
responses_multi = get_relevant_responses(multi_df)
with open('../data/relevant_khem.json', 'w', encoding ='utf8') as json_file:
    json.dump(responses_khem, json_file, ensure_ascii = False)
with open('../data/relevant_ragni.json', 'w', encoding ='utf8') as json_file:
    json.dump(responses_ragni, json_file, ensure_ascii = False)
with open('../data/relevant_multi.json', 'w', encoding ='utf8') as json_file:
    json.dump(responses_multi, json_file, ensure_ascii = False)
