#%%
# %load_ext autoreload
# %autoreload 2
# %%
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import utils
import numpy as np
# point to project folder
# droot = '/home/ulzee/imp/data'
droot = '/u/project/sriram/ulzee/imp/data'
ext = 'csv'
#%%
dname = 'mdd'
phase = 'test'
obs = 0.99
#%%
# method = '/home/ulzee/imp/data/andyv3pcs/OBS099_imputed_test_softimpute_.csv'
# method = f'/home/ulzee/imp/data/andyv3pcs/OBS0992_imputed_{phase}_deepcoder_RANDMASK03_LIVEMASK_WIDTH10_DEPTH1_MULT0_BATCH2048_RELU_SGDLR05M09_TESTMASK_VAL.csv'
# method = f'/home/ulzee/imp/data/andyv3pcs/OBS0992_imputed_{phase}_deepcoder_RANDMASK03_LIVEMASK_WIDTH10_DEPTH1_MULT0_BATCH2048_RELU_SGDLR05M09_TESTMASK_VAL.csv'
# method = '/home/ulzee/imp/data/andy/OBS099_imputed_test_softimpute_.csv'
# method = '/home/ulzee/imp/data/andy/OBS099_imputed_test_deepcoder_RANDMASK03_LIVEMASK_WIDTH10_DEPTH1_MULT0_BATCH2048_RELU_SGDLR01M09_VAL.csv'
# method = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test.csv'
method ='/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData.csv'
bootstrap = False
ext = 'csv'
pop = None
no_header = False
# %%
# print(sys.argv[1:])
# dname = sys.argv[1]
# phase = sys.argv[2]
# obs = float(sys.argv[3])
# method = sys.argv[4]
# bootstrap = len(sys.argv) >= 6 and sys.argv[5] == 'boot'
# if len(sys.argv) >= 7:
# 	ext = sys.argv[6]
# if len(sys.argv) >= 8:
# 	no_header = sys.argv[7] == 'no_header'
# pop = None
# if len(sys.argv) == 8:
# 	pop = int(sys.argv[7])
# 	print('Limit pop', pop)
#%%
dpath = f'{droot}/{dname}'
#%%
# method = 'softimpute'
use_sigmoid_methods = ['softimpute', 'gain']
cont_cats, binary_cats = utils.load_cats(dname, droot=droot)
#%%
all_cats = cont_cats + binary_cats
#%%
# obs = 0.95
# bootstrap = True
# method = 'hivae_1_z8_y10_s1_batch2048'
# ext = 'csv'
#%%
if phase == 'valonly':
	testfile = f'{dpath}/ptrain.tsv'
else:
	testfile = f'{dpath}/p{phase}.tsv'

vmat = utils.load_table(testfile)
vmat.shape
#%%
vmat
#%%
# tabFileName = f'OBS%03d_imputed_{phase}_{method}' % (100*obs)
# tabname = f'{dpath}/{tabFileName}.{ext}'
#%%
tabname = method
if no_header:
	imat = utils.load_table(tabname, no_header=True)
	print(imat.shape)
	imat.columns = vmat.columns
	if phase == 'valonly':
		imat['FID'] = vmat.iloc[-imat.shape[0]:].index
	else:
		imat['FID'] = vmat.index
	imat = imat.set_index('FID')
else:
	imat = utils.load_table(tabname)
imat.shape
#%%
if 'FID' not in imat.columns:
	if phase == 'valonly':
		imat['FID'] = vmat.iloc[-imat.shape[0]:].index
	else:
		imat['FID'] = vmat.index
	imat = imat.set_index('FID')
imat = imat.astype(float)
#%%
if imat.shape[1] < vmat.shape[1]:
	assert False
    # try to fix using colnames
	template = vmat.copy()
	template[imat.columns] = imat.values
	imat = template
	print('Fixed to:', imat.shape)
	# exit()
# %%
mask_tag = '0'
boot_tag = ''
# if 'UNIF' in method:
# 	mask_tag = 'unif'
# 	boot_tag = '_unif'
if obs <= 0.99:
	if phase == 'valonly':
		mask = np.load(f'{dpath}/masks/mask_train_OBS%03d_{mask_tag}.npy' % (100*obs))
	else:
		mask = np.load(f'{dpath}/masks/mask_{phase}_OBS%03d_{mask_tag}.npy' % (100*obs))
else:
	mask = np.load(f'{dpath}/masks/mask_{phase}_OBS%04d_{mask_tag}.npy' % (1000*obs)) # more observed
mask.shape
#%%
# pind = vmat.columns.tolist().index('LifetimeMDD')
# nscore = mask[:, pind] & ~vmat.iloc[:, pind].isna()
# nscore.sum()
#%%
mfrac = pd.read_csv(f'{dpath}/missing.csv').set_index('Unnamed: 0')
mfrac
#%%
if pop is not None:
	vmat = vmat.iloc[:pop,]
	imat = imat.iloc[:pop,]
	mask = mask[:pop,]
	print(vmat.shape)
	print(imat.shape)
	print(mask.shape)
# %%
debug = False
binary_only = ['pr', 'roc', 'bi']
need_pos_neg = ['pr', 'roc']
metrics = OrderedDict(
	mse=lambda a, b: np.mean((a-b)**2),
	r2=lambda a, b: np.corrcoef(a, b)[0, 1]**2,
	pr=utils.aucpr,
	roc=utils.aucroc,
)
#%%
if phase == 'valonly':
	vmat = vmat.iloc[-imat.shape[0]:]
	mask = mask[-imat.shape[0]:]
#%%
scores = OrderedDict(pheno=[], n=[])
fsd = lambda fname: f'{fname}.sd'
fci = lambda fname: f'{fname}.ci'
meta_stats=dict(
	nscore=[],
	balance=[],
	bootna=[],
	metric=[],
	pheno=[],
)
for fname, fn in metrics.items():
	scores[fname] = []
	scores[fsd(fname)] = []
	scores[fci(fname)] = []
values = []
skipped = []
nvals  = []
nanphenos = []
assets = dict(ypred=dict(), vtrue=dict())
bootmat = None
# debug = True
for pi in range(vmat.shape[1]):
	pname = vmat.columns[pi]
	if debug and pname != 'LifetimeMDD': continue
	score_v = mask[:, pi] & ~np.isnan(vmat.iloc[:, pi].values)
	nscore = np.sum(score_v)

	if nscore == 0:
		skipped += [pname]
		# if mfrac.loc[pname] != 0:
		# 	print(pname, '%.4f' % mfrac.loc[pname])
		continue
	nvals += [nscore]
	vtrue = vmat.iloc[:, pi].values[score_v].copy()
	ypred = imat.iloc[:, pi].values[score_v].copy()
	assets['vtrue'][pname] = vtrue
	assets['ypred'][pname] = ypred

	if pname == 'LifetimeMDD':
		# print(vtrue[:10])
		# print(ypred[:10])
		print(imat.iloc[:10, pi])
		print(score_v.sum())
	# if 'lefthanded.' in pname:
	# 	print(vtrue)
	# 	print(ypred)
	useSigmoid = pname in binary_cats and any([match in method for match in use_sigmoid_methods])
	# useSigmoid = True
	if useSigmoid:
		ypred = utils.sigmoid(ypred)
	if bootstrap:
		bootpath = f'{droot}/{dname}/boot'
		bootinds = np.load(
			f'{bootpath}/OBS%03d_P%03d_bootinds{boot_tag}.npy' % (100*obs, pi))
		if bootmat is None:
			bootmat = -np.ones((vmat.shape[1], len(bootinds)))
	values += [(score_v, vtrue, ypred)]

	replist = []
	for fname, fn in metrics.items():
		meta_stats['pheno'] += [pname]
		meta_stats['nscore'] += [nscore]
		meta_stats['balance'] += [None]
		meta_stats['bootna'] += [None]
		meta_stats['metric'] += [fname]
		if pname in binary_cats:
			meta_stats['balance'][-1] = np.sum(vtrue) / len(vtrue)

		val = np.nan
		valsd = np.nan
		valci = np.nan
		if pname in cont_cats and fname in binary_only:
			# some metrics not applicable to cont
			pass
		elif pname in binary_cats and len(np.unique(vtrue)) == 1:
			# in some cases pr,auc cannot be computed
			pass
		else:
			# if pname in binary_cats:
			# 	print(vtrue)
			val = fn(vtrue, ypred)
		scores[fname] += [val]

		if ~np.isnan(val) and bootstrap and np.all(bootinds != -1):
			reps = []
			for bi, bootset in enumerate(bootinds):
				__vtrue = vmat.iloc[:, pi].values[bootset].copy()
				__ypred = imat.iloc[:, pi].values[bootset].copy()
				if fname in need_pos_neg and utils.singular(__vtrue):
					# if fname == 'r2' and pname not in skipped:
						# skipped += [pname]
					continue
				bootval = fn(__vtrue, __ypred)
				reps += [bootval]
				if fn == metrics['r2']:
					bootmat[pi, bi] = bootval
			# assert len(reps) > 20 # FIXME: better way to handle bad reps
			if len(reps) > 10:
				# continue
				meta_stats['bootna'][-1] = np.sum(np.isnan(reps)) + (len(bootinds) - len(reps))
				# NOTE: some cases of few N, reps are same and sd is undefined
				valsd = np.std(reps, ddof=1)
				valci = np.mean(
					[abs(np.percentile(reps, 5) - val),abs(np.percentile(reps, 95) - val)])
				replist += [reps]
		scores[fsd(fname)] += [valsd]
		scores[fci(fname)] += [valci]

		if fname == 'r2' and np.isnan(val):
			nanphenos += [(pname, nscore, ypred, vtrue)]

	scores['pheno'] += [pname]
	scores['n'] += [int(nscore)]

	stringed = ' '.join(
		[f'{fname}:%.3f(%.2f)' % (
			scores[fname][-1], scores[fsd(fname)][-1]) for fname, _ in metrics.items()])
	print(f'[({pi+1}) {pname[:15]} : {nscore}] ', stringed)
#%%
# not_m1perc = mfrac.index[mfrac['0'] < 0.01]
m1perc = mfrac.index[mfrac['0'] >= 0.01]
#%%
template = pd.DataFrame(dict(pheno=vmat.columns.values))#.set_index('pheno', drop=False)
scoredf = pd.DataFrame(scores).set_index('pheno')
savedf = template.join(scoredf, on='pheno').set_index('pheno')
savedf
# %%
fname = tabname.split('/')[-1].split('.')[0]
saveName = f'score_{fname}.csv'
savedf.to_csv(f'{dpath}/{saveName}')
#%%
# naps = m1perc[savedf.loc[m1perc]['r2.ci'].isna()]
# mfrac.loc[naps]
# for phe in naps:
# 	cratio = (vmat[phe] == 1).sum() / (vmat[phe] == 0).sum()
# 	print(phe, vmat[phe].nunique())
#%%
# r2_ci_mean = savedf.loc[m1perc]['r2.ci'].mean()
# r2_est_mean = savedf.loc[m1perc]['r2'].mean()
# r2_est_mean, r2_ci_mean
# # %%
# m1perc_inds = [i for i, p in enumerate(vmat.columns) if p in m1perc]
# scored_bootmat = bootmat[m1perc_inds, :]
# valid_bootmat = scored_bootmat[np.isnan(scored_bootmat).sum(axis=1) == 0, :]
# valid_bootmat.shape
# # %%
# r2_boot_mean = valid_bootmat.mean(axis=0)
# hi = np.quantile(r2_boot_mean, 0.975)
# lo = np.quantile(r2_boot_mean, 0.025)
# '%.3f, (%.3f %.3f)' % (np.mean(r2_boot_mean), lo, hi)
# # %%
# fname = tabname.split('/')[-1].split('.')[0]
# bootSaveName = f'boot_{fname}.npy'
# np.save(f'{dpath}/{bootSaveName}', bootmat)
# bootSaveName
# # %%
