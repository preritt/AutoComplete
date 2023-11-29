#%%
import pandas as pd
import numpy as np
import argparse
import sys
sys.path.append('AutoComplete/')
import utils
#%%
droot = '/u/project/sriram/ulzee/imp/data'
ext = 'csv'
dname = 'mdd'
cont_cats, binary_cats = utils.load_cats(dname, droot=droot)
all_cats = cont_cats + binary_cats
# %%
import json
#%%  AutoComplete/datasets/allFeatureDataTransformer AutoComplete/datasets/allFeatureDataTransformerWithMask
# load the json file
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/results_r2.json') as json_file:
    data_ae_with_mask = json.load(json_file)
# %%
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/results_r2.json') as json_file:
    data_ae = json.load(json_file)
# %%
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataTransformer/results_r2.json') as json_file:
    data_transformer = json.load(json_file)
# %%
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataTransformerWithMask/results_r2.json') as json_file:
    data_transformer_with_mask = json.load(json_file)
# %%
continuous_metrics = data_ae[cont_cats[0]].keys()

data_ae_with_mask_continuous = {k: data_ae_with_mask[k] for k in cont_cats}
data_ae_continuous = {k: data_ae[k] for k in cont_cats}
data_transformer_with_mask_continuous = {k: data_transformer_with_mask[k] for k in cont_cats}
data_transformer_continuous = {k: data_transformer[k] for k in cont_cats}
# %%
mse_values_ae_with_mask_continuous  = [value["mse"] for value in data_ae_with_mask_continuous.values()]
mse_values_ae_continuous  = [value["mse"] for value in data_ae_continuous.values()]
mse_values_transformer_with_mask_continuous  = [value["mse"] for value in data_transformer_with_mask_continuous.values()]
mse_values_transformer_continuous  = [value["mse"] for value in data_transformer_continuous.values()]


r2_values_ae_with_mask_continuous  = [value["r2"] for value in data_ae_with_mask_continuous.values()]
r2_values_ae_continuous  = [value["r2"] for value in data_ae_continuous.values()]
r2_values_transformer_with_mask_continuous  = [value["r2"] for value in data_transformer_with_mask_continuous.values()]
r2_values_transformer_continuous  = [value["r2"] for value in data_transformer_continuous.values()]
# %%
float_indices = [index for index, value in enumerate(mse_values_ae_continuous) if isinstance(value, float)]
# %%
feature_names = [cont_cats[index] for index in float_indices]

# %%
mse_values_ae_with_mask_continuous_filtered = [mse_values_ae_with_mask_continuous[index] for index in float_indices]
mse_values_ae_continuous_filtered = [mse_values_ae_continuous[index] for index in float_indices]
mse_values_transformer_with_mask_continuous_filtered = [mse_values_transformer_with_mask_continuous[index] for index in float_indices]
mse_values_transformer_continuous_filtered = [mse_values_transformer_continuous[index] for index in float_indices]


r2_values_ae_with_mask_continuous_filtered = [r2_values_ae_with_mask_continuous[index] for index in float_indices]
r2_values_ae_continuous_filtered = [r2_values_ae_continuous[index] for index in float_indices]
r2_values_transformer_with_mask_continuous_filtered = [r2_values_transformer_with_mask_continuous[index] for index in float_indices]
r2_values_transformer_continuous_filtered = [r2_values_transformer_continuous[index] for index in float_indices]

# %%
# compute the mean mse values for all the cases
mse_values_ae_with_mask_continuous_filtered_mean = np.nanmean(mse_values_ae_with_mask_continuous_filtered)
mse_values_ae_continuous_filtered_mean = np.nanmean(mse_values_ae_continuous_filtered)
mse_values_transformer_with_mask_continuous_filtered_mean = np.nanmean(mse_values_transformer_with_mask_continuous_filtered)
mse_values_transformer_continuous_filtered_mean = np.nanmean(mse_values_transformer_continuous_filtered)
print('mse values ae with mask mean: ', mse_values_ae_with_mask_continuous_filtered_mean)
print('mse values ae mean: ', mse_values_ae_continuous_filtered_mean)
print('mse values transformer with mask mean: ', mse_values_transformer_with_mask_continuous_filtered_mean)
print('mse values transformer mean: ', mse_values_transformer_continuous_filtered_mean)
# %%
# compute the mean r2 values for all the cases
r2_values_ae_with_mask_continuous_filtered_mean = np.nanmean(r2_values_ae_with_mask_continuous_filtered)
r2_values_ae_continuous_filtered_mean = np.nanmean(r2_values_ae_continuous_filtered)
r2_values_transformer_with_mask_continuous_filtered_mean = np.nanmean(r2_values_transformer_with_mask_continuous_filtered)
r2_values_transformer_continuous_filtered_mean = np.nanmean(r2_values_transformer_continuous_filtered)
print('r2 values ae with mask mean: ', r2_values_ae_with_mask_continuous_filtered_mean)
print('r2 values ae mean: ', r2_values_ae_continuous_filtered_mean)
print('r2 values transformer with mask mean: ', r2_values_transformer_with_mask_continuous_filtered_mean)
print('r2 values transformer mean: ', r2_values_transformer_continuous_filtered_mean)
# %%
# %%
# make a bar plot showing a comparison between the two
import matplotlib.pyplot as plt
# %%
# Create an array of indices
indices = np.arange(len(feature_names))

# Set the width of the bars
bar_width = 0.35
plt.figure(figsize=(30,10))
# Create the bar plot
plt.bar(indices, mse_values_ae_with_mask_continuous_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, mse_values_ae_continuous_filtered, bar_width, label='Without Mask AC')
plt.bar(indices + bar_width*2, mse_values_transformer_with_mask_continuous_filtered, bar_width, label='Transformer With Mask')
plt.bar(indices + bar_width*3, mse_values_transformer_continuous_filtered, bar_width, label='Transformer Without Mask')

# Add labels and title to the plot
plt.xlabel('Features')
plt.ylabel('MSE Values')
plt.title('Comparison of MSE Values')
plt.xticks(indices + bar_width / 2, feature_names, rotation=45)
plt.legend()
# %%
plt.figure(figsize=(30,10))
# Create the bar plot
plt.bar(indices, r2_values_ae_with_mask_continuous_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, r2_values_ae_continuous_filtered, bar_width, label='Without Mask AC')
plt.bar(indices + bar_width*2, r2_values_transformer_with_mask_continuous_filtered, bar_width, label='Transformer With Mask')
plt.bar(indices + bar_width*3, r2_values_transformer_continuous_filtered, bar_width, label='Transformer Without Mask')



# Add labels and title to the plot
plt.xlabel('Features')
plt.ylabel('R2 Values')
plt.title('Comparison of R2 Values')
plt.xticks(indices + bar_width / 2, feature_names, rotation=45)
plt.legend()
# %%
binary_metrics = data_ae[binary_cats[0]].keys()
data_ae_with_mask_binary = {k: data_ae_with_mask[k] for k in binary_cats}
data_ae_binary = {k: data_ae[k] for k in binary_cats}
data_transformer_with_mask_binary = {k: data_transformer_with_mask[k] for k in binary_cats}
data_transformer_binary = {k: data_transformer[k] for k in binary_cats}
# %%
pr_values_ae_with_mask_binary  = [value["pr"] for value in data_ae_with_mask_binary.values()]
roc_values_ae_with_mask_binary  = [value["roc"] for value in data_ae_with_mask_binary.values()]

pr_values_ae_binary  = [value["pr"] for value in data_ae_binary.values()]
roc_values_ae_binary  = [value["roc"] for value in data_ae_binary.values()]

pr_values_transformer_with_mask_binary  = [value["pr"] for value in data_transformer_with_mask_binary.values()]
roc_values_transformer_with_mask_binary  = [value["roc"] for value in data_transformer_with_mask_binary.values()]

pr_values_transformer_binary  = [value["pr"] for value in data_transformer_binary.values()]
roc_values_transformer_binary  = [value["roc"] for value in data_transformer_binary.values()]
# %%
float_indices = [index for index, value in enumerate(pr_values_ae_binary) if isinstance(value, float)]
# %%
feature_names_binary = [binary_cats[index] for index in float_indices]

pr_values_ae_binary_filtered = [pr_values_ae_binary[index] for index in float_indices]
roc_values_ae_binary_filtered = [roc_values_ae_binary[index] for index in float_indices]

pr_values_ae_with_mask_binary_filtered = [pr_values_ae_with_mask_binary[index] for index in float_indices]
roc_values_ae_with_mask_binary_filtered = [roc_values_ae_with_mask_binary[index] for index in float_indices]

pr_values_transformer_binary_filtered = [pr_values_transformer_binary[index] for index in float_indices]
roc_values_transformer_binary_filtered = [roc_values_transformer_binary[index] for index in float_indices]

pr_values_transformer_with_mask_binary_filtered = [pr_values_transformer_with_mask_binary[index] for index in float_indices]
roc_values_transformer_with_mask_binary_filtered = [roc_values_transformer_with_mask_binary[index] for index in float_indices]

# %%
# compute the mean pr values for all the cases
pr_values_ae_with_mask_binary_filtered_mean = np.nanmean(pr_values_ae_with_mask_binary_filtered)
pr_values_ae_binary_filtered_mean = np.nanmean(pr_values_ae_binary_filtered)
pr_values_transformer_with_mask_binary_filtered_mean = np.nanmean(pr_values_transformer_with_mask_binary_filtered)
pr_values_transformer_binary_filtered_mean = np.nanmean(pr_values_transformer_binary_filtered)
print('pr values ae with mask mean: ', pr_values_ae_with_mask_binary_filtered_mean)
print('pr values ae mean: ', pr_values_ae_binary_filtered_mean)
print('pr values transformer with mask mean: ', pr_values_transformer_with_mask_binary_filtered_mean)
print('pr values transformer mean: ', pr_values_transformer_binary_filtered_mean)
# %%
# compute the mean roc values for all the cases
roc_values_ae_with_mask_binary_filtered_mean = np.nanmean(roc_values_ae_with_mask_binary_filtered)
roc_values_ae_binary_filtered_mean = np.nanmean(roc_values_ae_binary_filtered)
roc_values_transformer_with_mask_binary_filtered_mean = np.nanmean(roc_values_transformer_with_mask_binary_filtered)
roc_values_transformer_binary_filtered_mean = np.nanmean(roc_values_transformer_binary_filtered)
print('roc values ae with mask mean: ', roc_values_ae_with_mask_binary_filtered_mean)
print('roc values ae mean: ', roc_values_ae_binary_filtered_mean)
print('roc values transformer with mask mean: ', roc_values_transformer_with_mask_binary_filtered_mean)
print('roc values transformer mean: ', roc_values_transformer_binary_filtered_mean)
# %%
# Create an array of indices
plt.figure(figsize=(30,10))
indices = np.arange(len(feature_names_binary))
plt.bar(indices, pr_values_ae_with_mask_binary_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, pr_values_ae_binary_filtered, bar_width, label='Without Mask (AC)')
plt.bar(indices + bar_width*2, pr_values_transformer_with_mask_binary_filtered, bar_width, label='Transformer With Mask')
plt.bar(indices + bar_width*3, pr_values_transformer_binary_filtered, bar_width, label='Transformer Without Mask')

plt.xlabel('Features')
plt.ylabel('PR Values')
plt.title('Comparison of PR Values')
plt.xticks(indices + bar_width / 2, feature_names_binary, rotation=45)
plt.legend()
# %%
plt.figure(figsize=(30,10))
indices = np.arange(len(feature_names_binary))
# plt.bar(indices, roc_values_ae_with_mask_binary_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, roc_values_ae_binary_filtered, bar_width, label='Without Mask (AC)')
# plt.bar(indices + bar_width*2, roc_values_transformer_with_mask_binary_filtered, bar_width, label='Transformer With Mask')
plt.bar(indices + bar_width*3, roc_values_transformer_binary_filtered, bar_width, label='Transformer Without Mask')
plt.xlabel('Features')
plt.ylabel('ROC Values')
plt.title('Comparison of ROC Values')
plt.xticks(indices + bar_width / 2, feature_names_binary, rotation=45)
plt.legend()
# %%
