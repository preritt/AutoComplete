#%% 
# %load_ext autoreload
# %autoreload 2
#%% AutoComplete/datasets/allFeatureDataTransformer   AutoComplete/datasets/allFeatureDataTransformerWithMask
import pandas as pd
from time import time
import json
import argparse
import sys
#%%
class args:
    # data_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureDataTransformer/ptrain.csv'
    data_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptrain.csv'
    id_name = 'FID'
    lr = 0.01
    batch_size = 1024
    val_split = 0.8
    device = 'cuda:0'
    epochs = 50
    momentum = 0.9
    # impute_using_saved = 'datasets/mate_male/data_fit.pth'
    impute_using_saved = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptrain.pth'
    output = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/data_fit_imputed_AEWithMAskOrigFeatUlzee_test_allFeatureData_0p1_p.csv'
    encoding_ratio = 1
    depth = 1
    impute_data_file = '/u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datasets/allFeatureData/ptest.csv'
    copymask_amount = 0.5
    num_torch_threads = 8
    simulate_missing = 0.01
    bootstrap = False
    seed = -1
    quality = False
    multiple = -1
    save_model_path = None
    save_imputed = True
#%%
# parser = argparse.ArgumentParser(description='AutoComplete')
# parser.add_argument('data_file', type=str, help='CSV file where rows are samples and columns correspond to features.')
# parser.add_argument('--id_name', type=str, default='ID', help='Column in CSV file which is the identifier for the samples.')
# parser.add_argument('--output', type=str, help='The imputed version of the data will be saved as this file. ' +\
#     'If not specified the imputed data will be saved as `imputed_{data_file}` in the same folder as the `data_file`.')

# parser.add_argument('--save_model_path', type=str, help='A location to save the imputation model weights. Will default to file_name.pth if not set.', default=None)

# parser.add_argument('--copymask_amount', type=float, default=0.3, help='Probability that a sample will be copy-masked. A range from 10%%~50%% is recommemded.')
# parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for fitting the model.')
# parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
# parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for fitting the model. A starting LR between 2~0.1 is recommended.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default is recommended).')
# parser.add_argument('--val_split', type=float, default=0.8, help='Amount of data to use as a validation split. The validation split is monitored for convergeance.')
# parser.add_argument('--device', type=str, default='cpu:0', help='Device available for torch (use cpu:0 if no GPU available).')
# parser.add_argument('--encoding_ratio', type=float, default=1,
#     help='Size of the centermost encoding dimension as a ratio of # of input features; ' + \
#     'eg. `0.5` would force an encoding by half.')
# parser.add_argument('--depth', type=int, default=1, help='# of fully connected layers between input and centermost deep layer; ' + \
#     'the # of layers beteen the centermost layer and the output layer will be defined equally.')

# parser.add_argument('--save_imputed', help='Will save an imputed version of the matrix immediately after fitting it.', action='store_true', default=False)
# parser.add_argument('--impute_using_saved', type=str, help='Load trained weights from a saved .pth file to ' + \
#     'impute the data without going through model training.')
# parser.add_argument('--impute_data_file', type=str, help='CSV file where rows are samples and columns correspond to features.')
# parser.add_argument('--seed', type=int, help='A specific seed to use. Can be used to instantiate multiple imputations.', default=-1)
# parser.add_argument('--bootstrap', help='Flag to specify whether the dataset should be bootstrapped for the purpose of fitting.', default=False, action='store_true')
# parser.add_argument('--multiple', type=int, help='If set, this script will save a list of commands which can be run (either in sequence or in parallel) to save mulitple imputations', default=-1)
# parser.add_argument('--quality', help='Applies to the fitting procedure. If set, this script will compute a variance ratio metric and a r^2 metric for each feature to roughly inform the quality of imputation', default=False, action='store_true')
# parser.add_argument('--simulate_missing', help='Specifies the %% of original data to be simulated as missing for r^2 computation.', default=0.01, type=float)
# parser.add_argument('--num_torch_threads', help='Prevents torch from taking up all threads on a device. Can be increased when only running one fit but default can be sufficient.', default=8, type=int)

# args = parser.parse_args()
#%%
if args.multiple != -1:
    print('Saving commands for multiple imputations based on the current configs.')
    configs = sys.argv[1:]
    mi = configs.index('--multiple')
    configs.pop(mi)
    configs.pop(mi)
    with open('multiple_imputation.sh', 'w') as fl:
        fl.write('\n'.join([
            'python fit.py ' + ' '.join(configs) + f' --seed {m} --bootstrap --save_imputed'
            for m in range(args.multiple)]))
    exit()
#%%
fparts = args.data_file.split('/')
save_folder = '/'.join(fparts[:-1]) + '/'
filename = args.data_file.split('/')[-1].replace('.csv', '')
save_model_path = save_folder + filename

if args.output:
    save_table_name = args.output
else:
    save_table_name = save_folder + f'imputed_{filename}'

if args.seed != -1:
    save_table_name += f'_seed{args.seed}'
    save_model_path += f'_seed{args.seed}'
if args.bootstrap:
    save_table_name += f'_bootstrap'
    save_model_path += f'_bootstrap'

save_model_path += '.pth'
if not args.output: save_table_name += '.csv'

if args.save_model_path is not None:
    save_model_path = args.save_model_path

if not args.impute_using_saved:
    print('Saving model to:', save_model_path)
if args.impute_using_saved or args.save_imputed:
    print('Saving imputed table to:', save_table_name)
#%%
import sys
sys.path.append('..')
#%%
import torch
torch.set_num_threads(args.num_torch_threads)
import random
import numpy as np

if args.seed != -1:
    print(f'Using seed: {args.seed}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ac import AutoComplete
from ac import AutoCompleteWithMissingMask
from ac import TransformerNoPosAutoCompleteWithoutMissingMask
from ac import TransformerNoPosAutoCompleteWithMissingMask
from dataset import CopymaskDataset
from datasetTest import TestDataset as Test
#%%
tab = pd.read_csv(args.data_file).set_index(args.id_name)
test_data = pd.read_csv(args.impute_data_file).set_index(args.id_name)
print(f'Dataset size:', tab.shape[0])
print(f'Dataset size for test:', test_data.shape[0])
# test_data_mask_file = pd.read_csv('/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv').set_index(args.id_name)
test_data_mask_file = pd.read_csv('/u/project/sriram/ulzee/imp/data/mdd/masks/mask_test_OBS099_0.csv').set_index(args.id_name)


#%%
if args.bootstrap:
    print('Bootstrap mode')
    ix = list(range(len(tab)))
    ix = np.random.choice(ix, size=len(tab), replace=True)
    tab = tab.iloc[ix]
    print('First few ids are:')
    for i in tab.index[:5]:
        print(' ', i)

#%%
# detect binary phenotypes
ncats = tab.nunique()
binary_features = tab.columns[ncats == 2]
contin_features = tab.columns[~(ncats == 2)]
feature_ord = list(contin_features) + list(binary_features)
print(f'Features loaded: contin={len(contin_features)}, binary={len(binary_features)}')
CONT_BINARY_SPLIT = len(contin_features)
#%%
index_mdd = feature_ord.index('MDDRecur')
print(index_mdd)
# %%
# keep a validation set
val_ind = int(tab.shape[0]*args.val_split)
splits = ['train', 'val', 'final', 'test']
dsets = dict(
    train=tab[feature_ord].iloc[:val_ind, :],
    val=tab[feature_ord].iloc[val_ind:, :],
    final=tab[feature_ord],
    test = test_data[feature_ord]
)
dset_test = test_data[feature_ord]
# %%
# train_stats = dict(
#     mean=dsets['train'].mean().values,
#     std=dsets['train'].std().values,
# )
train_stats = dict(mean=dsets['train'].mean().values)
train_stats['std'] = np.nanstd(dsets['train'].values - train_stats['mean'], axis=0)
#%%
normd_dsets = {
    split: (dsets[split].values - train_stats['mean'])/train_stats['std'] \
        for split in splits }
# %%
# dataloaders = {
#     split: torch.utils.data.DataLoader(
#         CopymaskDataset(mat, split, copymask_amount=args.copymask_amount),
#         batch_size=args.batch_size,
#         shuffle=split=='train', num_workers=0) \
#             for split, mat in normd_dsets.items() }
# dataloaders = {
#     split: torch.utils.data.DataLoader(
#         Test(mat, split, copymask_amount=args.copymask_amount),
#         batch_size=args.batch_size,
#         shuffle=split=='train', num_workers=0) \
#             for split, mat in normd_dsets.items() }
# dataloaders = {
#     split: torch.utils.data.DataLoader(
#         Test(normd_dsets['test'], split, copymask_amount=args.copymask_amount),
#         batch_size=args.batch_size,
#         shuffle=False, num_workers=0) \
#             for split, mat in normd_dsets.items() }
dataloaders = {}

for split, mat in normd_dsets.items():
    if split != 'test':
        dataloaders[split] = torch.utils.data.DataLoader(
            CopymaskDataset(normd_dsets[split], split, copymask_amount=args.copymask_amount),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
    )
    else:
        dataloaders[split] = torch.utils.data.DataLoader(
            Test(normd_dsets[split], split, test_data_mask_file.values, copymask_amount=args.copymask_amount),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
    )

#%%
feature_dim = dsets['train'].shape[1]
# core = AutoComplete(
#         indim=feature_dim,
#         width=1/args.encoding_ratio,
#         n_depth=args.depth,
#     )
# core = AutoCompleteWithMissingMask(
#         indim=feature_dim,
#         width=1/args.encoding_ratio,
#         n_depth=args.depth,
#     )
core = AutoCompleteWithMissingMask(
        indim=feature_dim,
        width=1/args.encoding_ratio,
        n_depth=args.depth,
    )

# core = TransformerNoPosAutoCompleteWithoutMissingMask(
#         indim=feature_dim,
#     )
# core = TransformerNoPosAutoCompleteWithMissingMask(
#         indim=feature_dim,
#     )
model = core.to(args.device)
print('Model name is :', model.__class__.__name__)
#%%
# Define a function to compute the gradient of the ith output with respect to the inputs

# def compute_gradient(model, datarow, output_number=10):
#     # Set the model to evaluation mode
#     model.eval()

#     # Convert the datarow to a PyTorch tensor
#     # datarow_tensor = torch.tensor(datarow, dtype=torch.float32)
#     datarow_tensor = datarow.clone().detach()

#     # Enable gradient computation for the inputs
#     datarow_tensor.requires_grad = True

#     # Perform the forward pass
#     output = model(datarow_tensor)

#     # Select the specified output
#     output_selected = output[:, output_number]

#     # Compute the gradients of each element in the output_selected tensor with respect to the inputs
#     gradients = []
#     for i in range(len(output_selected)):
#         gradient = torch.autograd.grad(output_selected[i], datarow_tensor[i], retain_graph=True)[0]
#         gradients.append(gradient)

#     return gradients

#%%
def compute_gradient(model, datarow, output_number):
    # Set model to evaluation mode
    model.eval()

    # Convert inputs to PyTorch tensor
    # inputs_tensor = torch.tensor(datarow, requires_grad=True)
    # inputs_tensor = torch.tensor(datarow).clone().detach().requires_grad_(True)
    inputs_tensor = datarow.clone().detach().requires_grad_(True)


    # Forward pass to get the output
    outputs = model(inputs_tensor)

    # Extract the specified output
    selected_output = outputs[:, output_number]

    # Backward pass to compute gradients
    selected_output.backward(torch.ones_like(selected_output))

    # Get the gradients of the inputs
    gradients = inputs_tensor.grad

    return gradients.detach()
#%%
if args.impute_using_saved:
    print(f'Loading specified weights: {args.impute_using_saved}')
    model = torch.load(args.impute_using_saved)

if (args.save_imputed or args.quality) and not args.impute_using_saved:
    print('Loading last best checkpoint')
    model = torch.load(save_model_path)

if args.impute_data_file or args.save_imputed or args.quality:
    model = model.to(args.device)
    model.eval()

    impute_mat = args.impute_data_file if args.impute_data_file else args.data_file
    imptab = pd.read_csv(impute_mat).set_index(args.id_name)[feature_ord]
    print(f'(impute) Dataset size:', imptab.shape[0])

    mat_imptab = (imptab.values - train_stats['mean'])/train_stats['std']


    dset = torch.utils.data.DataLoader(
        Test(mat_imptab, 'final', test_data_mask_file.values),
        batch_size=args.batch_size,
        shuffle=False, num_workers=0)    

    preds_ls = []
    all_gradients = []
    all_gradients_mdd_unknown = []
    print('Model name is :', model.__class__.__name__)
    for bi, batch in enumerate(dset):
        datarow, _, masked_inds = batch
        datarow = datarow.float().to(args.device)

        if args.quality:
            # sim_mask = sim_missing[bi*args.batch_size:(bi+1)*args.batch_size]
            # PT - updated to use the simulated nans from the test data
            sim_mask = masked_inds
            datarow[sim_mask] = 0

        with torch.no_grad():
            yhat = model(datarow)
        # compute the gradient of the ith output with respect to the inputs
        gradients = compute_gradient(model, datarow, output_number=index_mdd)
        completed_frame = yhat.clone().detach()
        # make the index_mdd column to 0
        completed_frame[:, index_mdd] = 0
        gradients_mdd_all_known = compute_gradient(model, completed_frame, output_number=index_mdd)

        all_gradients+=[gradients.cpu().numpy()] 
        all_gradients_mdd_unknown+=[gradients_mdd_all_known.cpu().numpy()]   
        sind = CONT_BINARY_SPLIT
        yhat = torch.cat([yhat[:, :sind], torch.sigmoid(yhat[:, sind:])], dim=1)

        preds_ls += [yhat.cpu().numpy()]
        print(f'\rImputing: {bi}/{len(dset)}', end='')

    pmat = np.concatenate(preds_ls)
    pmat *= train_stats['std']
    pmat += train_stats['mean']
    all_gradients_np = np.concatenate(all_gradients)
    all_gradients_mdd_unknown_np = np.concatenate(all_gradients_mdd_unknown)
    print()


    if args.impute_data_file or args.save_imputed:
        template = imptab.copy()
        tmat = template.values
        tmat[test_data_mask_file.values==1] = pmat[test_data_mask_file.values==1]
        # tmat[np.isnan(tmat)] = pmat[np.isnan(tmat)]
        
        template[:] = tmat
        template

        template.to_csv(save_table_name)
print('Model name is :', model.__class__.__name__)
print('done')

# %%
