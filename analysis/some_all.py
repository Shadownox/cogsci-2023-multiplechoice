import pandas as pd
import numpy as np
df = pd.read_csv("../data/multiple_choice.csv")

print("Some entails All:", np.mean(df.groupby("id").agg("first")["some_entails_all"].values))

print("Total number of participants:", len(np.unique(df["id"])))
