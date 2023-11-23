#%% 

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the first CSV file
data_AE = pd.read_csv('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/ptrain_quality.csv')

# Load the data from the second CSV file
data_AEExtended = pd.read_csv('/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptrain_quality.csv')

# Extract the relevant columns from the first file
features1 = data_AE['feature']
r2_scores1 = data_AE['r2']
quality_metrics1 = data_AE['quality']

# Extract the relevant columns from the second file
features2 = data_AEExtended['feature']
r2_scores2 = data_AEExtended['r2']
quality_metrics2 = data_AEExtended['quality']

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = range(len(features1))
r2 = [x + bar_width for x in r1]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(r1, r2_scores1, color='b', width=bar_width, label='File 1')
plt.bar(r2, r2_scores2, color='g', width=bar_width, label='File 2')
plt.xlabel('Feature')
plt.ylabel('Metric')
plt.title('Comparison of Metrics between File 1 and File 2')
plt.xticks([r + bar_width/2 for r in range(len(features1))], features1, rotation=90)
plt.legend()
plt.show()
# %%
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(r1, quality_metrics1, color='b', width=bar_width, label='File 1')
plt.bar(r2, quality_metrics2, color='g', width=bar_width, label='File 2')
plt.xlabel('Feature')
plt.ylabel('Metric')
plt.title('Comparison of Metrics between File 1 and File 2')
plt.xticks([r + bar_width/2 for r in range(len(features1))], features1, rotation=90)
plt.legend()
plt.show()
# %%
