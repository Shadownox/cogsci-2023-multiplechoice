import pandas as pd
import numpy as np
import ccobra
import json
from scipy.stats import mannwhitneyu

response_list = [x.lower() for x in ccobra.syllogistic.RESPONSES]

def gen_task_enc(elem):
    item = ccobra.Item(0, "syllogistic", elem, "single-choice", "", 0)
    syl = ccobra.syllogistic.Syllogism(item)
    return syl.encoded_task

def gen_resp_enc(elem):
    item = ccobra.Item(0, "syllogistic", elem["task"], "single-choice", "", 0)
    syl = ccobra.syllogistic.Syllogism(item)
    return "['{}']".format(syl.encode_response(elem["response"].split(";")).lower())

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

# Load khemlani2012
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

# Load ragni2016
ragni_df = pd.read_csv("../data/Ragni2016.csv")
ragni_df["enc_task"] = ragni_df["task"].apply(gen_task_enc)
ragni_df["enc_responses"] = ragni_df[["task", "response"]].apply(gen_resp_enc, axis=1)
ragni_mat = get_mat(ragni_df)

# Load multiple_choice
multi_df = pd.read_csv("../data/multiple_choice.csv")
multi_mat = get_mat(multi_df)

# Calculate figural effect
def figure_diff(mat):
    figure1_syllogs = [(x, y) for x, y in enumerate(ccobra.syllogistic.SYLLOGISMS) if y[2] == "1"]
    figure1_responses = [(x, y) for x, y in enumerate(ccobra.syllogistic.RESPONSES) if y[1:] == "ac"]
    
    figure2_syllogs = [(x, y) for x, y in enumerate(ccobra.syllogistic.SYLLOGISMS) if y[2] == "2"]
    figure2_responses = [(x, y) for x, y in enumerate(ccobra.syllogistic.RESPONSES) if y[1:] == "ca"]
    
    effect_votes = []
    contra_votes = []
    figure1_diffs = []
    for syl_idx, syllog in figure1_syllogs:
        fig1_votes = np.sum([mat[syl_idx, x] for x, y in figure1_responses])
        fig2_votes = np.sum([mat[syl_idx, x] for x, y in figure2_responses])
        figure1_diffs.append(fig1_votes - fig2_votes)
        effect_votes.append(fig1_votes)
        contra_votes.append(fig2_votes)

    figure2_diffs = []
    for syl_idx, syllog in figure2_syllogs:
        fig1_votes = np.sum([mat[syl_idx, x] for x, y in figure1_responses])
        fig2_votes = np.sum([mat[syl_idx, x] for x, y in figure2_responses])
        figure2_diffs.append(fig2_votes - fig1_votes)
        effect_votes.append(fig2_votes)
        contra_votes.append(fig1_votes)
        
    return figure1_diffs, figure2_diffs, effect_votes, contra_votes

def analyze_figure_effect(mat):
    figure1_diffs, figure2_diffs, effect_votes, contra_votes = figure_diff(mat)
    total_diff = figure1_diffs + figure2_diffs
    print("    total differences:\t\tmean={},\tSD={}".format(np.round(np.mean(total_diff), 2), np.round(np.std(total_diff), 2)))
    print("    figure 1 differences:\tmean={},\tSD={}".format(np.round(np.mean(figure1_diffs), 2), np.round(np.std(figure1_diffs), 2)))
    print("    figure 2 differences:\tmean={},\tSD={}".format(np.round(np.mean(figure2_diffs), 2), np.round(np.std(figure2_diffs), 2)))
    U, p = mannwhitneyu(effect_votes, contra_votes, method="exact")
    print("    Mann-Whitney-U:\t\tU={},\tp={}".format(U, p))


def check_implicature(df, precondition, implication):
    occurances = 0
    total = 0
    for _, row in df.iterrows():
        responses = row["enc_responses"]
        responses = eval(responses)
        responses_quant = set([x[0] for x in responses if x != "nvc"])

        if precondition in responses_quant:
            total += 1
            if implication in responses_quant:
                occurances += 1
    return occurances/total
    
print("Implicatures")
print("A implies I: {}".format(np.round(check_implicature(multi_df, "a", "i"),3)))
print("E implies O: {}".format(np.round(check_implicature(multi_df, "e", "o"),3)))
print("I implies O: {}".format(np.round(check_implicature(multi_df, "i", "o"),3)))
print("O implies I: {}".format(np.round(check_implicature(multi_df, "o", "i"),3)))
print()

print("Figure effect")
print("Multiple Choice")
analyze_figure_effect(multi_mat)
print()

print("Ragni2016 vs Multiple Choice")
r1, r2, _, _ = figure_diff(ragni_mat)
m1, m2, _, _ = figure_diff(multi_mat)
ragni_diffs = r1 + r2
multi_diffs = m1 + m2
U, p = mannwhitneyu(ragni_diffs, multi_diffs, method="exact")
print("    Differences:\t\tRagni={}\tMulti={}".format(np.round(np.mean(ragni_diffs),2), np.round(np.mean(multi_diffs),2)))
print("    Mann-Whitney-U:\t\tU={},\tp={}".format(U, p))
print()
