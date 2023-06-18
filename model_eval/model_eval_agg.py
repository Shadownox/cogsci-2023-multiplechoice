import json
import ccobra
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, basename
import scipy.stats

def jaccard_sim(data, model):
    if not isinstance(data, set):
        data = set(data)
    if not isinstance(model, set):
        model = set(model)
        
    return len(data.intersection(model)) / len(data.union(model))
    
def calc_hits(data, model):
    if not isinstance(data, set):
        data = set(data)
    if not isinstance(model, set):
        model = set(model)
        
    return len(data.intersection(model)) / len(data)

def calc_rejections(data, model):  
    rejections = set([x for x in ccobra.syllogistic.RESPONSES if x not in data])
    model_rejections = set([x for x in ccobra.syllogistic.RESPONSES if x not in model])
    
    return len(rejections.intersection(model_rejections)) / len(rejections)

def calc_correct(data, model):
    cor = 0
    for resp in ccobra.syllogistic.RESPONSES:
        if resp not in data and resp not in model:
            cor += 1
        elif resp in data and resp in model:
            cor += 1
    return cor / len(ccobra.syllogistic.RESPONSES)

khem = None
ragni = None
multi = None

with open('../data/relevant_khem.json', 'r', encoding ='utf8') as json_file:
    khem = json.load(json_file)
with open('../data/relevant_ragni.json', 'r', encoding ='utf8') as json_file:
    ragni = json.load(json_file)
with open('../data/relevant_multi.json', 'r', encoding ='utf8') as json_file:
    multi = json.load(json_file)

models = [join("models", file) for file in listdir("models") if isfile(join("models", file)) and file.endswith("csv")]

model_rej = []
model_num = []
model_hits = []

for model in models:
    model_df = pd.read_csv(model)
    print("{}".format(model[7:-4]))
    
    total_rej = []
    total_num = []
    total_hits = []
    
    for data_name, dataset in [("MultipleChoice", multi), ("Ragni2016\t", ragni), ("Khemlani2012", khem)]:
        jaccards = []
        hits = []
        rejections = []
        correct = []
        num_responses = []

        for syllog in ccobra.syllogistic.SYLLOGISMS:
            data_pred = dataset[syllog]
            pred = model_df[model_df["Syllogism"] == syllog]["Prediction"].values[0].split(";")
            
            num_responses.append(len(pred))
            jaccards.append(jaccard_sim(data_pred, pred))
            hits.append(calc_hits(data_pred, pred))
            rejections.append(calc_rejections(data_pred, pred))
            correct.append(calc_correct(data_pred, pred))

        jaccards = np.round(np.mean(jaccards), 2)
        hits = np.round(np.mean(hits), 2)
        rejections = np.round(np.mean(rejections), 2)
        correct = np.round(np.mean(correct), 2)
        num_responses = np.round(np.mean(num_responses), 2)
        total_rej.append(rejections)
        total_hits.append(hits)
        total_num.append(num_responses)

        print("    {}:\tcorrect={}, jaccard={}, hits={}, rejections={}, num_responses={} ".format(data_name, correct, jaccards, hits, rejections, num_responses))
    print()
    model_rej.append(np.mean(total_rej))
    model_num.append(np.mean(total_num))
    model_hits.append(np.mean(total_hits))
    
print("Spearman (Model rejection - Model num responses): ", scipy.stats.spearmanr(model_rej, model_num))