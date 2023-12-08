#%%
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
v0df = pd.read_csv('/u/project/sriram/ulzee/imp/data/mdd/autocomplete_paper_OBS099_scores.csv').set_index('pheno')
v0df
#%%
# with open('/u/scratch/z/zhengton/AutoComplete/datasets/AEWithMask/results_r2.json') as fl:
# 	v1 = json.load(fl)
# %%
# /u/scratch/z/zhengton/AutoComplete/datasets/AE_with_mask_depth_3/results_r2.json
#%%
# dr = '/u/scratch/z/zhengton/AutoComplete/datasets/AEWithMask/results_r2.json'
# dr = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataTransformerV2/results_r2_v3.json'
# dr =  '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/transformerIncorrectWithMask/results_r2_v3.json'
dr ='/u/scratch/z/zhengton/AutoComplete/datasets/Transformer_batch_as_seq/results_r2.json'
# allFeatureDataTransformerV2
with open(dr) as fl:
	v1 = json.load(fl)
# %%
phenos = v0df.index.intersection(list(v1.keys()))
len(phenos)
# %%
plt.figure(figsize=(5, 5))
plt.title('Transformer (Incorrect) - Batch 16')
plt.scatter(v0df.loc[phenos]['est'].values, [v1[ph]['r2'] for ph in phenos])
plt.scatter(v0df.loc['LifetimeMDD']['est'], [v1[ph]['r2'] for ph in ['LifetimeMDD']])
plt.xlabel('Original AC')
plt.ylabel('AC with Mask concat')
plt.axline((0,0), slope=1, color='gray')
plt.show()
np.nanmean(v0df.loc[phenos]['est'].values), np.nanmean([v1[ph]['r2'] for ph in phenos])
# %%
