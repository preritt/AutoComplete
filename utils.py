
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
import os, sys

def load_table(fname, no_header=False):
	# if 'csv' in fname:
	sep = ','

	if 'tsv' in fname:
		sep = '\t'
	# NOTE: hotfix
	# if 'softimpute' in fname:
	# 	sep = '\t'

	if no_header:
		df = pd.read_csv(fname, dtype=float, sep=sep, header=None)
	else:
		df = pd.read_csv(fname, dtype=float, sep=sep)
		if 'FID' in df.columns:
			df = df.set_index('FID')
	return df

def load_cats(dname, droot='/u/scratch/u/ulzee/imp/data'):
	catfile = f'{droot}/{dname}/cats.csv'
	cats = pd.read_csv(catfile)

	return cats[cats['isbinary'] == False]['cats'].values.tolist(), \
		cats[cats['isbinary'] == True]['cats'].values.tolist()

def singular(ls):
	return len(np.unique(ls)) == 1

def aucpr(trues, preds):
	if singular(trues): return None
	pr, re, _ = metrics.precision_recall_curve(trues, preds)
	return metrics.auc(re, pr)

def aucroc(trues, preds):
	if singular(trues): return None
	fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
	return metrics.auc(fpr, tpr)

# def biserial(trues, preds):
# 	rval, _ = stats.pointbiserialr(trues, preds)
# 	return rval

sigmoid = lambda v: 1 / (1 + np.e**-v)

def get_common_phenos(metric, bymethod, incats=None, nlimit=100, obs_check=None, methods_check=None, cats=None):
	notna = None
	methods_check = methods_check if methods_check is not None else list(bymethod.keys())
	for method in methods_check:
		tab = bymethod[method]
		# print(tab)
		if cats is not None:
			tab = tab.loc[cats]
		obslist = np.unique(tab['obs'].values) if obs_check is None else obs_check
		for obs in obslist:
			obstab = tab[(tab['obs'] == obs)]
			# if obstab.shape[0] == 0:
			# 	return None
			if obstab.shape[0] > 0:
				if notna is None: notna = ~obstab[metric].isna()
				else: notna = notna & ~obstab[metric].isna()

	enoughN = (tab[(tab['obs'] == obs)]['n'] >= nlimit).values
	notna &= enoughN
	# print(notna)
	common_phenos = tab[tab['obs'] == 0.99].index[notna].tolist()
	if incats is not None:
		common_phenos = [c for c in common_phenos if c in incats]
	# print(tab.loc[common_phenos][metric].isna().sum().sum())
	return common_phenos

def default_name_mapping(
	name_mapping = {
		'deepcoder_RANDMASK05_WIDTH10_DEPTH1_MULT0_BATCH2048_RELU_SGDLR20M09': 'AutoComplete',
		'softimpute': 'SoftImpute',
		'hivae_1_z8_y10_s1_batch2048': 'HI-VAE',
		'gain_HINT09_ALPHA0100_BATCH2048_EPOCHS5000': 'GAIN',
		'knn': 'KNN',
	}):
	return name_mapping

def load_scores(
	dname,
	show_methods,
	custom_load_byobs={},
	show_obs=[0.99, 0.95, 0.90, 0.80, 0.50],
	droot='/u/scratch/u/ulzee/imp/data',
	prefix='score2', phase='test'):

	dpath = f'{droot}/{dname}'

	bymethod = dict()
	for method in show_methods:
		tabs = []
		load_name = method
		for oi, obs in enumerate(show_obs):
			if method in custom_load_byobs:
				load_name = custom_load_byobs[method][oi]
			tabName = f'{prefix}_OBS%03d_imputed_{phase}_{load_name}.csv' % (100*obs)
			fullTabName = f'{dpath}/{tabName}'
			if os.path.exists(fullTabName):
				tab = pd.read_csv(fullTabName, index_col='pheno')
				tab['obs'] = obs
				tabs += [tab]
			else:
				print('Not found:', fullTabName)
		bymethod[method] = pd.concat(tabs)
	return bymethod