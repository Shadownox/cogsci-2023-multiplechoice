import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse, Circle
import seaborn as sns
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
    
def plot_pattern(ax, df, title, mat=None, weighted=False):
    mat, mfp_dict = get_mat(df, weighted=weighted)
    mat = mat.T

    sns.heatmap(mat, ax=ax, cmap="Blues", cbar=False, vmin=0, linewidths=0.5, linecolor='#00000022')
    ax.set_yticks(np.arange(len(ccobra.syllogistic.RESPONSES)) + 0.5)
    ax.set_yticklabels(ccobra.syllogistic.RESPONSES, rotation=0)
    ax.set_xticks(np.arange(len(ccobra.syllogistic.SYLLOGISMS), step=4) + 0.7)
    ax.set_xticklabels(ccobra.syllogistic.SYLLOGISMS[::4], rotation=90)
    ax.tick_params(axis='x', pad=-4)
        
    colors = ["red", "purple"]
    for syl_idx, syl in enumerate(ccobra.syllogistic.SYLLOGISMS):
        mfps = mfp_dict[syl]
        for color_idx, mfp in enumerate(mfps):
            responses = mfp[0]
            for response in responses:
                resp_idx = response_list.index(response)
                ax.add_patch(Ellipse((syl_idx + 0.5, resp_idx + 0.5),0.4,0.3, fill=True, facecolor=colors[color_idx], edgecolor=colors[color_idx], lw=0.3))


sns.set(style='whitegrid', palette='colorblind')

fig, ax = plt.subplots(1, 1, figsize=(10, 2.5), sharex=True)

plot_pattern(ax, multi_df, "Multiple Choice (unweighted)", weighted=False)

plt.tight_layout()
plt.savefig("multi_pattern.pdf", bbox_inches='tight', pad_inches = 0)
plt.show()
