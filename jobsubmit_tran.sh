#!/bin/sh
#$ -cwd
#$ -l h_data=16G,h_rt=00:59:00,gpu
#$ -o /u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datalogs
#$ -e /u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete/datalogs
# Email address to notify
#$ -M $pterway@ucla.edu
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

echo '/usr/bin/time -v hostname'

source /u/local/Modules/default/init/modules.sh
module load cuda
module load anaconda3/2023.03 
# conda activate /u/local/apps/anaconda3/2023.03/envs/pytorch-2.0-gpu
conda activate /u/home/p/pterway/project-sriram/condaEnvironment/uclaTransformer/envs/llmenvac
# Source the .bashrc file
# mamba init
# source ~/.bashrc
# which python
# conda activate llmenvac
# which python
# mamba env list
# conda activate /u/scratch/p/pterway/condaEnv/llmEnvScratch
cd /u/scratch/p/pterway/UCLAProjects/ulzeeAutocomplete/AutoComplete
#gbasePath='/u/project/sgss/UKBB/data/wes/plink-23155/'
#bannotPath='/u/scratch/b/boyang19/Angela/gene/'

#annotPath=("High-HDL_genes_forRanges.txt" "High-LDL_genes_forRanges.txt" "High-triglycerides_genes_forRanges.txt" "MODY_genes_forRanges.txt" "Low-LDL_genes_forRanges.txt" "Obesity_genes_forRanges.txt")

# genepath='/u/project/sgss/UKBB/data/wes/plink-23155/ukb23155'
# annotPath='/u/scratch/b/boyang19/Angela/gene/High-HDL_genes_forRanges.txt'
python fit.py
# for i in ${!annotPath[@]};
# do
#     python fcreate.py $genepath ${bannotPath}${annotPath[$i]}
# done