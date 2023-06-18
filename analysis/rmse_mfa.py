import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse, Circle
import seaborn as sns
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

def get_mfa_mat(mat):
    max_vals = np.max(mat, axis=1, keepdims=True)
    return mat == max_vals

def mfa_congruence(A, B):
    match = A == B
    return np.mean(match)

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

# Calculate MFA congruence
print("MFA congruence")
khem_mfa = get_mfa_mat(khemlani_mat)
ragni_mfa = get_mfa_mat(ragni_mat)
multi_mfa = get_mfa_mat(multi_mat)

print("Khemlani2012 - Ragni2016:\tmfac={}".format(mfa_congruence(khem_mfa, ragni_mfa)))
print("Khemlani2012 - Multiple Choice:\tmfac={}".format(mfa_congruence(khem_mfa, multi_mfa)))
print("Ragni2016 - Multiple Choice:\tmfac={}".format(mfa_congruence(ragni_mfa, multi_mfa)))
print()

# Calculate rmse
print("RMSE")
def mse(A, B):
    mse = (np.square(A - B)).mean()
    return mse
mse_khem_ragni = mse(khemlani_mat, ragni_mat)
mse_khem_multi = mse(khemlani_mat, multi_mat)
mse_ragni_multi = mse(ragni_mat, multi_mat)

print("Khemlani2012 - Ragni2016:\tmse={}, rmse={}".format(mse_khem_ragni, np.sqrt(mse_khem_ragni)))
print("Khemlani2012 - Multiple Choice:\tmse={}, rmse={}".format(mse_khem_multi, np.sqrt(mse_khem_multi)))
print("Ragni2016 - Multiple Choice:\tmse={}, rmse={}".format(mse_ragni_multi, np.sqrt(mse_ragni_multi)))

