# Purpose: Convert tsv file to csv file
#%%
import csv

tsv_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptest.tsv'
csv_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptest.csv'

with open(tsv_file, 'r') as tsvfile, open(csv_file, 'w', newline='') as csvfile:
    tsv_reader = csv.reader(tsvfile, delimiter='\t')
    csv_writer = csv.writer(csvfile)

    for row in tsv_reader:
        csv_writer.writerow(row)
# %%
