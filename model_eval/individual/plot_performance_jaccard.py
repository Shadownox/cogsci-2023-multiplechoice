import pandas as pd
import numpy as np
import ccobra
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CCOBRA evaluation data
result_df = pd.read_csv("results/jaccard.csv")

result_df = result_df[result_df["model"].isin(["PHM (indiv)", "PHM", "MFA", "MMT", "mReasoner"])]

# Normalize model names
result_df['model'] = result_df['model'].replace({
    #'VerbalModels': 'Verbal\nModels',
    'PHM (indiv)': 'PHM\n(indiv)'
    #'PHM (NVC)': 'PHM\n(NVC)'
})

# Mean the data
subj_df = result_df.groupby(
    ['model', 'id'], as_index=False)['score_response'].agg('mean')

order_df = subj_df.groupby(['model'], as_index=False)['score_response'].agg('mean')
order = order_df.sort_values('score_response')['model']

# Prepare for plotting
sns.set(style="whitegrid", palette='colorblind')
plt.figure(figsize=(4, 2.5))

# Color definition
point = [0.01, 0.3, 0.6]
box = [0.01, 0.3, 0.6, 0.5]

# Plot the data
sns.swarmplot(x="model", y="score_response", data=subj_df, order=order,
              dodge=True, linewidth=0.5, size=2, edgecolor=[0.3,0.3,0.3], color=point, zorder=1)

ax = sns.boxplot(x="model", y="score_response", data=subj_df, order=order,
                 showcaps=False,boxprops={'facecolor': box, "zorder":10},
                 showfliers=False,whiskerprops={"zorder":10}, linewidth=1, color="black",
                 zorder=10, showmeans=True, meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})

plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.xlabel('')
ax.set_ylabel('Jaccard Coefficient', size=12)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('coverage_swarmplot_jacc.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()
