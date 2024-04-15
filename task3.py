######## TASK 3 - LAMBDAMART########################
################## ###################################
######################################################################

# Importing the necessary libraries
import json
import pandas as pd
import numpy as np
from termcolor import colored
from tqdm import tqdm
import re  
import spacy  
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser']) # for tokenization
from rank_bm25 import BM25Okapi # for BM25
import os
from collections import defaultdict 
import scipy as sp
from scipy import spatial
from scipy import stats
from scipy.optimize import fmin
import ast 
import multiprocessing
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
import xgboost as xgb
import sklearn


####################### MAKE OUTPUT DIRECTORY ############################
outdir = './outputs/task3'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dir_task2 = './outputs/task2'
############# LOADING THE COSINE SIMILARITY FROM TASK 2############################
training_data_sample = pd.read_csv(dir_task2 + '/training_data_sample_sim.txt', sep=',')
validation_data = pd.read_csv(dir_task2+ '/validation_data_sim.txt', sep=',')
candidate_passages = pd.read_csv(dir_task2 + '/candidate_passages_sim.txt', sep=',')
############### SORTING THE DATA BY QID ########################################
training_data_sample= training_data_sample.sort_values(by='qid')
############### QIDS

train_qid = training_data_sample['qid']
validation_qid = validation_data['qid']
test_qid = candidate_passages['qid']

######################FUNCTIONS ###############################################
################################################################################
#################### BM25 IMPLEMENTATION ########################################
def bm25_feature(clean_df, qids):
    bm25_scores = np.array([])
    for qid in tqdm(qids):
        docs = clean_df[clean_df["qid"]==qid].reset_index(drop=True) #candidate rows for this query
        query = docs["clean queries"][0] #current query
        corpus = [passage for passage in docs["clean passages"]]
        bm25 = BM25Okapi(corpus) #initialize bm25
        # Find BM25 score for the query
        query = clean_df[clean_df["qid"]==qid]["clean queries"].reset_index(drop=True)[0]
        bm25_scores = np.concatenate([bm25_scores, bm25.get_scores(query)])
    return bm25_scores


################## FUNCTIONS FOR PRECISION AND NDCG ###############
# Mean precision takes top 100 ranks for each query and calculates the precision   
def mean_precision(df, validation_qid):
    precision_per_qid = []
    retrieved = []
    for qid in validation_qid:
        top100 = df[df['qid']==qid]['rank'].head(100).index
        overlap_count = df[(df['relevancy']==1) & (df.index.isin(top100))].shape[0]
        precision_per_qid.append(overlap_count/len(top100))
        retrieved.append(len(top100))
    return np.mean(precision_per_qid), retrieved

def mean_ndcg(df, retrieved):
        # Calculate NDCG
    # requires dataframe to be sorted by qid
    df.sort_values(by='qid', inplace=True)
    DCG = []
    IDCG = []
    NDCG = []
    #retrieved also counts as num of ranks
    #go through each qid in df and calculate DCG: filter out 0s first i
    for element in retrieved:
        IDCG.append(sum([1/np.log2(i + 1) for i in range(1, element + 1)]))   
    done_qid = []
    d= 0
    index = 0
    counter = 0
    for ind, row in tqdm(df[df['relevancy']==1].iterrows(), total=df[df['relevancy']==1].shape[0]):
        i = row['rank']
        if row['qid'] not in done_qid and counter>0: # for new query (unless its the first one), append DCG of previous query 
            done_qid.append(row['qid'])
            DCG.append(d) #append previous 
            NDCG.append(d/IDCG[index])
            index += 1
            d = 0 #reset
            d += 1/np.log2(i+2)
        else:
            d += 1/np.log2(i+2)
        counter += 1
    return np.mean(NDCG)
#################### FEATURES: BM25 AND COSINE SIM ######################
################## CALCULATE BM25 FEATURES ############################
training_data_sample['bm25'] = bm25_feature(training_data_sample, train_qid)
validation_data['bm25'] = bm25_feature(validation_data, validation_qid)
candidate_passages['bm25'] = bm25_feature(candidate_passages, test_qid)


# Features
X_training = training_data_sample[['bm25', 'sim']].values 
y_training = training_data_sample['relevancy'].values

############## LAMBDAMART IMPLEMENTATION ###############################
def lambda_mart(X_train, y_train, train_qid, k, obj):
    objective_name = f"rank:{obj}"
    ranker = xgb.XGBRanker(tree_method="hist",
                            lambdarank_num_pair_per_sample=k,
                            objective=objective_name,
                            lambdarank_pair_method="topk")
    ranker.fit(X_train, y_train, qid=train_qid)
    path = f"{outdir}/XGBRanker_{obj}_{k}.model"
    ranker.save_model(path)
    return ranker
############################ VALIDATION DATA ############################

# try objectives rank:ndcg and rank:map because align with target metrics
# use k top ranked documents for each query. try 3 different k values
# k = 5, 10, 20
k = [5, 10, 20]
objective = ['ndcg', 'map']
#######################################################################
# for each k value and objective, train a ranker and evaluate on validation data
################################################################
for k_val in k:
    for obj in objective:
        ranker = lambda_mart(X_training, y_training, train_qid, k_val, obj)
        validation_data[f"score_{k_val}_{obj}"] = ranker.predict(validation_data[["bm25", "sim"]].values)
        validation_data = validation_data.sort_values(by=["qid", f"score_{k_val}_{obj}"])
        validation_data['rank'] = validation_data.groupby('qid')[f"score_{k_val}_{obj}"].rank(ascending=False, method='first')
        validation_data['rank'] = validation_data['rank'].astype(int)
        mean_precision_logi, retrieved_validation_logi = mean_precision(validation_data, validation_qid)
        mean_ndcg_logi = mean_ndcg(validation_data, retrieved_validation_logi)
        # sSAVE METRICS
        with open(outdir + '/metrics_lambda.txt', 'a') as f:
            f.write(f"For k_val = {k_val} and obj = {obj}:\n")
            f.write(f"Mean Precision Logistic: {mean_precision_logi}\n")
            f.write(f"Mean NDCG Logistic: {mean_ndcg_logi}\n\n")
        print(f"For k_val = {k_val} and obj = {obj}:")
        print(f"Mean Precision Logistic: {mean_precision_logi}")
        print(f"Mean NDCG Logistic: {mean_ndcg_logi}")
        

# Select the best model based on the best NDGC and MAP
# # 10 map has best performance

##################################### RERANK CANDIDATE PASSAGES ########################
# Load the best model
ranker = xgb.XGBRanker()
k = 10
objective = 'map'
# Load the model from the file
path = f"{outdir}/XGBRanker_{objective}_{k}.model"
ranker.load_model(path)
# Predict the scores for the candidate passages
candidate_passages["score_10_map"] = ranker.predict(candidate_passages[["bm25", "sim"]].values)
# Rerank
candidate_passages['rank'] = candidate_passages.groupby('qid')['score_10_map'].rank(ascending=False, method='first')
candidate_passages['rank'] = candidate_passages['rank'].astype(int)
test_qid = candidate_passages['qid'].unique()
# Need to retrieve only at most 100 documents per query
# Save the top 100 documents for each query
candidate_passages_copy = candidate_passages.copy()
top = pd.DataFrame()
for qid in test_qid:
    top100 = candidate_passages_copy[candidate_passages_copy['qid'] == qid].sort_values(by='rank').head(100)
    top = pd.concat([top, top100], axis=0)
top['A2'] = 'A2'
top['algoname'] = 'LM'  
top = top[['qid', 'A2', 'pid', 'rank', 'score_10_map', 'algoname']]
# save to outdir
top.to_csv(outdir + '/LM.txt', sep=' ', index=False, header=False)