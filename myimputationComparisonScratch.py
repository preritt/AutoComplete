#%%
import pandas as pd
import utils
from collections import OrderedDict
import numpy as np
# %%
# load the predicted file which is a csv file
# filename = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData.csv'
filename='/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData_50p.csv'

# load the csv file
dataset_predicted = pd.read_csv(filename)
# set the index name to be the first column
dataset_predicted.index.name = dataset_predicted.columns[0]
# %%
# ensure that there is no nans in the prediction using assertion
assert dataset_predicted.isnull().values.any() == False
# %%
# load the original test file
filename_test = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptest.csv'
# load the csv file
dataset_test = pd.read_csv(filename_test)
# set the index name to be the first column
dataset_test.index.name = dataset_test.columns[0]

# %%
obs = 0.99
# %%
# load the file with masks which is a csv file
phase = 'test'
# mask_filename = '/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv'
mask_filename ='/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS050_0.csv'
# mask_filename ='/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData_50p.csv'
# mask_filename = '/u/project/sriram/ulzee/imp/data/mdd/masks/mask_{phase}_OBS%03d_{mask_tag}.npy' % (100*obs)

# load the csv file
dataset_mask = pd.read_csv(mask_filename)
# set the index name to be the first column
dataset_mask.index.name = dataset_mask.columns[0]
# %%
# rearrange the rows of the mask file to match the order of the test file
# dataset_mask = dataset_mask.reindex(dataset_test.index)
# %%
# extract the column names from the test file
column_names = list(dataset_test.columns.values)
# %%
dname = 'mdd'
droot = '/u/project/sriram/ulzee/imp/data'
use_sigmoid_methods = ['softimpute', 'gain']
cont_cats, binary_cats = utils.load_cats(dname, droot=droot)
all_cats = cont_cats + binary_cats

# %%
binary_only = ['pr', 'roc', 'bi']
need_pos_neg = ['pr', 'roc']
metrics = OrderedDict(
	mse=lambda a, b: np.mean((a-b)**2),
	r2=lambda a, b: np.corrcoef(a, b)[0, 1]**2,
	pr=utils.aucpr,
	roc=utils.aucroc,
)
mse1=lambda a, b: np.mean((a-b)**2)
# %%
# loop through the column names except the first column which is the index
for column_name in column_names[1:]:
    # extract the column from the predicted file
    column_predicted = dataset_predicted[column_name]
    # extract the column from the test file
    column_test = dataset_test[column_name]
    # extract the column from the mask file
    column_mask = dataset_mask[column_name]
    # %%
    # check if the column is a binary column
    # score indices are the indices where the mask is true and the test column is not nan
    score_indices = column_mask & ~column_test.isna()
    # %%
    # extract the score indices from the predicted column
    column_predicted_score = column_predicted[score_indices]
    # assert there are no nans in the predicted column
    assert column_predicted_score.isnull().values.any() == False
    # extract the score indices from the test column
    column_test_score = column_test[score_indices]
    # assert there are no nans in the test column
    assert column_test_score.isnull().values.any() == False
    # %%
    for fname, fn in metrics.items():
        print(fname)
        # calculate the score using the function
        score = fn(column_test_score, column_predicted_score)
        
        print(score)
        # append the score to the scores dictionary
        # scores[fname].append(score)
        # # append the score to the values list
        # values.append(score)

# %%
