import json
import ccobra
import numpy as np

def jaccard_sim(set1, set2):
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
        
    return len(set1.intersection(set2)) / len(set1.union(set2))

def num_intersection(set1, set2):
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    
    return len(set1.intersection(set2))

def compare_datasets(data1, data2, metric=jaccard_sim, agg=np.mean):
    results = []
    for syl in ccobra.syllogistic.SYLLOGISMS:
        responses1 = data1[syl]
        responses2 = data2[syl]
        sim = metric(responses1, responses2)
        results.append(sim)
    return agg(results)

khem = None
ragni = None
multi = None

with open('../data/relevant_khem.json', 'r', encoding ='utf8') as json_file:
    khem = json.load(json_file)
with open('../data/relevant_ragni.json', 'r', encoding ='utf8') as json_file:
    ragni = json.load(json_file)
with open('../data/relevant_multi.json', 'r', encoding ='utf8') as json_file:
    multi = json.load(json_file)

print("Jaccard Similarity:")
khem_ragni = compare_datasets(khem, ragni, metric=jaccard_sim)
khem_multi = compare_datasets(khem, multi, metric=jaccard_sim)
ragni_multi = compare_datasets(ragni, multi, metric=jaccard_sim)

print("Free Choice - Single Choice:\t\t", khem_ragni)
print("Free Choice - Multiple Choice:\t\t", khem_multi)
print("Single Choice - Multiple Choice:\t", ragni_multi)

print()

print("Avg number of responses:")
print("Free Choice:\t\t\t\t", compare_datasets(khem, khem, metric=num_intersection))
print("Single Choice:\t\t\t\t", compare_datasets(ragni, ragni, metric=num_intersection))
print("Multiple Choice:\t\t\t", compare_datasets(multi, multi, metric=num_intersection))
print()


print("Number of intersections:")
khem_ragni = compare_datasets(khem, ragni, metric=num_intersection)
khem_ragni_min = compare_datasets(khem, ragni, metric=num_intersection, agg=np.min)

khem_multi = compare_datasets(khem, multi, metric=num_intersection)
khem_multi_min = compare_datasets(khem, multi, metric=num_intersection, agg=np.min)

ragni_multi = compare_datasets(ragni, multi, metric=num_intersection)
ragni_multi_min = compare_datasets(ragni, multi, metric=num_intersection, agg=np.min)

print("Free Choice - Single Choice:\t\tmean={},\tmin={}".format(khem_ragni, khem_ragni_min))
print("Free Choice - Multiple Choice:\t\tmean={},\tmin={}".format(khem_multi, khem_multi_min))
print("Single Choice - Multi:\t\t\tmean={},\tmin={}".format(ragni_multi, ragni_multi_min))

print()