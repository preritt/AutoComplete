#%%
import pandas as pd
import numpy as np
import argparse
import sys
sys.path.append('AutoComplete/')
#%% AutoComplete/datasets/allFeatureData  AutoComplete/datasets/allFeatureDataAEOnly AutoComplete/datasets/allFeatureDataTransformer AutoComplete/datasets/allFeatureDataTransformerWithMask

# AutoComplete/datasets/allFeatureDataTransformerWithMask  AutoComplete/datasets/allFeatureDataAEOnly
class args:
	data_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/ptest.csv'
	simulated_data_file = 'datasets/phenotypes/data_test.csv'
	imputed_data_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData_0p1_p_v2.csv'
	mask_data_file = '/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv'
	num_bootstraps = 100
	saveas = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataAEOnly/results_r2_v2.json'
#%%
# parser = argparse.ArgumentParser(description='AutoComplete')
# parser.add_argument('data_file', type=str, help='Ground truth data. CSV file where rows are samples and columns correspond to features.')
# parser.add_argument('--simulated_data_file', type=str, help='Data with simulated missing values. This is required to check which values were simulated as missing.')
# parser.add_argument('--imputed_data_file', type=str, help='Imputed data.')
# parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of times to bootstrap the test statistic.')
# parser.add_argument('--saveas', type=str, default='results_r2.csv', help='Where to save the evaluation results.')
# args = parser.parse_args()

# In[]
import pandas as pd
import utils
from collections import OrderedDict
import numpy as np
# %%
# load the predicted file which is a csv file
# filename = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData.csv'
# filename='/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData.csv'

# load the csv file
dataset_predicted = pd.read_csv(args.imputed_data_file)
# set the index name to be the first column
dataset_predicted.index.name = dataset_predicted.columns[0]
# %%
# ensure that there is no nans in the prediction using assertion
assert dataset_predicted.isnull().values.any() == False
# %%
# load the original test file
# filename_test = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptest.csv'
# load the csv file
dataset_test = pd.read_csv(args.data_file)
# set the index name to be the first column
dataset_test.index.name = dataset_test.columns[0]

# %%
obs = 0.99
# %%
# load the file with masks which is a csv file
phase = 'test'
# mask_filename = '/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv'
# mask_filename ='/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv'
# mask_filename ='/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData_50p.csv'
# mask_filename = '/u/project/sriram/ulzee/imp/data/mdd/masks/mask_{phase}_OBS%03d_{mask_tag}.npy' % (100*obs)

# load the csv file
dataset_mask = pd.read_csv(args.mask_data_file)
# set the index name to be the first column
dataset_mask.index.name = dataset_mask.columns[0]
# %%
# replace all the nans in dataset_test with 0
dataset_test_copy = dataset_test.copy()
dataset_test_copy['FID'] = dataset_test_copy['FID'].astype(int)


# dataset_test_copy = dataset_test_copy.fillna(0)
# make a copy of the predicted data	
dataset_predicted_copy = dataset_predicted.copy()
dataset_predicted_copy['FID'] = dataset_predicted_copy['FID'].astype(int)

# make a copy of the mask data
dataset_mask_copy = dataset_mask.copy()
dataset_mask_copy['FID'] = dataset_mask_copy['FID'].astype(int)

# %%
# create a 

# %%
# rearrange the rows of the mask file to match the order of the test file
# dataset_mask = dataset_mask.reindex(dataset_test.index)
# %%
# extract the column names from the test file
column_names = list(dataset_test.columns.values)
#%%
# the original dataset
# original_data = pd.read_csv(args.data_file).set_index('ID')
original_data = dataset_test_copy.copy()

original_data
#%%
# data with simulated missing values
# simulated_data = pd.read_csv(args.simulated_data_file).set_index('ID')
simulated_data = dataset_mask_copy.copy()

simulated_data
#%%
# imputed_data = pd.read_csv(args.imputed_data_file).set_index('ID')
imputed_data = dataset_predicted_copy.copy()
imputed_data
#%%
assert simulated_data.shape == imputed_data.shape
assert simulated_data.index.tolist() == imputed_data.index.tolist()
assert imputed_data.isna().sum().sum() == 0
assert len(imputed_data.index.intersection(original_data.index)) == len(imputed_data)
#%%
metrics_continuous = OrderedDict(
	mse=lambda a, b: np.mean((a-b)**2),
	r2=lambda a, b: np.corrcoef(a, b)[0, 1]**2,
)
metrics_discrete = OrderedDict(
	mse=lambda a, b: np.mean((a-b)**2),
	r2=lambda a, b: np.corrcoef(a, b)[0, 1]**2,
	pr=utils.aucpr,
	roc=utils.aucroc,
)
scores_continuous = OrderedDict(pheno=[], n=[])
for fname, fn in metrics_continuous.items():
	scores_continuous[fname] = []

scores_discrete = OrderedDict(pheno=[], n=[])
for fname, fn in metrics_discrete.items():
	scores_discrete[fname] = []
#%%
use_sigmoid_methods = ['softimpute', 'gain']
droot = '/u/project/sriram/ulzee/imp/data'
ext = 'csv'
dname = 'mdd'
cont_cats, binary_cats = utils.load_cats(dname, droot=droot)
all_cats = cont_cats + binary_cats

#%%
mse=lambda a, b: np.mean((a-b)**2)
ests = []
stds = []
nsize = len(imputed_data)
corr = {}
mse_track = {}
all_scores = {}
for pheno in all_cats:
	all_scores[pheno] = {}
	if pheno in cont_cats:
		for fname, fn in metrics_continuous.items():
			all_scores[pheno][fname] = []
			pass
	elif pheno in binary_cats:
		for fname, fn in metrics_discrete.items():
			all_scores[pheno][fname] = []
			pass
#%%
for pheno in all_cats:
	print(pheno)
	similated_mask_locations = simulated_data[pheno]
	original_data_mask_locations = original_data[pheno]
	nan_locations_orig_data = pd.isna(original_data_mask_locations).astype(int)
	# find locations where original_data_mask_locations 
	indices_where_predicted = np.where((nan_locations_orig_data == 0) & (similated_mask_locations == 1))[0]
	missing_frac = len(indices_where_predicted)/nsize
	if pheno in cont_cats:
		metrics = metrics_continuous
	elif pheno in binary_cats:
		metrics = metrics_discrete
	else:
		raise ValueError(f'Unknown phenotype: {pheno}')
	# for fname, fn in metrics.items():
	# 	all_scores[pheno][fname] = []
	# 	pass



	if missing_frac != 0:
		predictions_at_missing_locations = imputed_data[pheno].iloc[indices_where_predicted]
		true_values_at_missing_locations = original_data[pheno].iloc[indices_where_predicted]
		if pheno in binary_cats:
			predictions_at_missing_locations = utils.sigmoid(predictions_at_missing_locations)
			# convert true_values_at_missing_locations to binary
			# true_values_at_missing_locations = true_values_at_missing_locations.astype(int)

		for fname, fn in metrics.items():
			print(fname)
			all_scores[pheno][fname] = fn(true_values_at_missing_locations.values, predictions_at_missing_locations.values)
		


		# r2 = np.corrcoef(
		# 	predictions_at_missing_locations.values,
		# 	true_values_at_missing_locations.values)[0, 1]**2
		# print(r2)
		# mean_sq_err = mse(predictions_at_missing_locations.values, true_values_at_missing_locations.values)
		# # corr.append(r2)
		# corr[pheno] = r2
		# mse_track[pheno] = mean_sq_err
	else:
		print(f'{pheno} ({missing_frac*100:.1f}%)')
		corr[pheno] = 'NA'
		mse_track[pheno] = 'NA'
		for fname, fn in metrics.items():
			all_scores[pheno][fname] = 'NA'
	# missing_frac = simulated_data[pheno].isna().sum() / nsize

	# est = np.nan
	# stderr = np.nan
	# if missing_frac != 0:
	# 	stats = []
	# 	# for n in range(args.num_bootstraps):
	# 	n = 0
	# 	while n < args.num_bootstraps:
	# 		boot_idx = np.random.choice(range(nsize), size=nsize, replace=True)
	# 		boot_obs = original_data.loc[imputed_data.index][pheno].iloc[boot_idx]
	# 		boot_imp = imputed_data[pheno].iloc[boot_idx]

	# 		simulated_missing_inds = simulated_data[pheno].iloc[boot_idx].isna() & ~boot_obs.isna()

	# 		if simulated_missing_inds.sum() == 0:
	# 			continue

	# 		r2 = np.corrcoef(
	# 			boot_obs.values[simulated_missing_inds],
	# 			boot_imp.values[simulated_missing_inds])[0, 1]**2

	# 		n += 1
	# 		stats += [r2]
	# 	est = np.nanmean(stats)
	# 	stderr = np.nanstd(stats)
	# 	print(f'{pheno} ({missing_frac*100:.1f}%): {est:.4f} ({stderr:.4f})')
	# else:
	# 	print(f'{pheno} ({missing_frac*100:.1f}%)')

	# ests += [est]
	# stds += [stderr]
#%%
# Save the dictionary as a json file
import json
# Open the file in write mode and save the dictionary as JSON
with open(args.saveas, "w") as json_file:
    json.dump(all_scores, json_file)
# %%
# results = pd.DataFrame(dict(pheno=imputed_data.columns, estimates=ests, stderrs=stds)).set_index('pheno')
# results.to_csv(args.saveas)
# results
# %%
