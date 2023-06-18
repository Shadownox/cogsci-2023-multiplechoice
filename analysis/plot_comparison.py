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

def plot_pattern(ax, df, title, mat=None, show_relevant=True, weighted=False, show_labels=False):
    if mat is None:
        mat = get_mat(df, weighted=weighted)
    mfa_mat = None
    if df is None:
        mfa_mat = mat
    else:
        mfa_mat = get_mat(df, weighted=weighted)
    mat = mat.T

    sns.heatmap(mat, ax=ax, cmap="Blues", cbar=False, vmin=0, linewidths=0.5, linecolor='#00000022')
    ax.set_yticks(np.arange(len(ccobra.syllogistic.RESPONSES)) + 0.5)
    ax.set_yticklabels(ccobra.syllogistic.RESPONSES, rotation=0)
    
    if show_labels:
        ax.set_xticks(np.arange(len(ccobra.syllogistic.SYLLOGISMS), step=4) + 0.6)
        ax.set_xticklabels(ccobra.syllogistic.SYLLOGISMS[::4], rotation=90)
        ax.tick_params(axis='x', pad=-4)
    else:
        ax.set_xticklabels([])
    rectangle_offset = 0.0001
    if show_relevant:
        for y in range(64):
            for x in range(9):
                if mat[x, y] >= 0.16:
                    ax.add_patch(Ellipse((y + 0.5, x + 0.5),0.35,0.3, fill=True, facecolor='red', edgecolor='red', lw=0.3))
                
    maxes = np.argmax(mfa_mat, axis=1)
    for row, position in enumerate(maxes):
        
        ax.add_patch(Rectangle((row+rectangle_offset, position+rectangle_offset),(1-2*rectangle_offset), (1-2*rectangle_offset), clip_on=False, fill=False, edgecolor="black", lw=0.9))

# Plot the heatmap
sns.set(style='whitegrid', palette='colorblind')

fig, axs = plt.subplots(3, 1, figsize=(10, 5.5), sharey=True)

plot_pattern(axs[0], multi_df, "Multiple Choice (weighted)", show_relevant=True, weighted=True)
plot_pattern(axs[1], orig_df, "Single Choice (Ragni2016)")
plot_pattern(axs[2], None, "Free Response (Khemlani2012)", mat=khemlani_mat, show_labels=True)

axs[0].set_title("Multiple Choice", rotation=90, x=-0.06, y=0.125)
axs[1].set_title("Single Choice", rotation=90, x=-0.06, y=0.15)
axs[2].set_title("Free Response", rotation=90, x=-0.06, y=0.1)

plt.subplots_adjust(hspace=0)

plt.tight_layout()
plt.savefig("comparison_heatmap_weighted.pdf", bbox_inches='tight', pad_inches = 0)
plt.show()

