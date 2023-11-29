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
#%%
# load the json file
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/results_r2.json') as json_file:
    data_ae_with_mask = json.load(json_file)
# %%
with open('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/results_r2.json') as json_file:
    data_ae = json.load(json_file)

# %%
continuous_metrics = data_ae[cont_cats[0]].keys()

data_ae_with_mask_continuous = {k: data_ae_with_mask[k] for k in cont_cats}
data_ae_continuous = {k: data_ae[k] for k in cont_cats}
# %%
mse_values_ae_with_mask_continuous  = [value["mse"] for value in data_ae_with_mask_continuous.values()]
mse_values_ae_continuous  = [value["mse"] for value in data_ae_continuous.values()]


r2_values_ae_with_mask_continuous  = [value["r2"] for value in data_ae_with_mask_continuous.values()]
r2_values_ae_continuous  = [value["r2"] for value in data_ae_continuous.values()]
# %%
float_indices = [index for index, value in enumerate(mse_values_ae_continuous) if isinstance(value, float)]
# %%
feature_names = [cont_cats[index] for index in float_indices]

# %%
mse_values_ae_with_mask_continuous_filtered = [mse_values_ae_with_mask_continuous[index] for index in float_indices]
mse_values_ae_continuous_filtered = [mse_values_ae_continuous[index] for index in float_indices]

r2_values_ae_with_mask_continuous_filtered = [r2_values_ae_with_mask_continuous[index] for index in float_indices]
r2_values_ae_continuous_filtered = [r2_values_ae_continuous[index] for index in float_indices]

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
plt.bar(indices + bar_width, mse_values_ae_continuous_filtered, bar_width, label='Without Mask')

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
plt.bar(indices + bar_width, r2_values_ae_continuous_filtered, bar_width, label='Without Mask')

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
# %%
pr_values_ae_with_mask_binary  = [value["pr"] for value in data_ae_with_mask_binary.values()]
roc_values_ae_with_mask_binary  = [value["roc"] for value in data_ae_with_mask_binary.values()]

pr_values_ae_binary  = [value["pr"] for value in data_ae_binary.values()]
roc_values_ae_binary  = [value["roc"] for value in data_ae_binary.values()]
# %%
float_indices = [index for index, value in enumerate(pr_values_ae_binary) if isinstance(value, float)]
# %%
feature_names_binary = [binary_cats[index] for index in float_indices]

pr_values_ae_binary_filtered = [pr_values_ae_binary[index] for index in float_indices]
roc_values_ae_binary_filtered = [roc_values_ae_binary[index] for index in float_indices]

pr_values_ae_with_mask_binary_filtered = [pr_values_ae_with_mask_binary[index] for index in float_indices]
roc_values_ae_with_mask_binary_filtered = [roc_values_ae_with_mask_binary[index] for index in float_indices]
# %%
# Create an array of indices
plt.figure(figsize=(30,10))
indices = np.arange(len(feature_names_binary))
plt.bar(indices, pr_values_ae_with_mask_binary_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, pr_values_ae_binary_filtered, bar_width, label='Without Mask')

plt.xlabel('Features')
plt.ylabel('PR Values')
plt.title('Comparison of PR Values')
plt.xticks(indices + bar_width / 2, feature_names_binary, rotation=45)
plt.legend()
# %%
plt.figure(figsize=(30,10))
indices = np.arange(len(feature_names_binary))
plt.bar(indices, roc_values_ae_with_mask_binary_filtered, bar_width, label='With Mask')
plt.bar(indices + bar_width, roc_values_ae_binary_filtered, bar_width, label='Without Mask')
plt.xlabel('Features')
plt.ylabel('ROC Values')
plt.title('Comparison of ROC Values')
plt.xticks(indices + bar_width / 2, feature_names_binary, rotation=45)
plt.legend()
# %%
