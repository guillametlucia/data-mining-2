######## TASK 1 - EVALUATING RETRIEVAL QUALITY #######################
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
import multiprocessing
cores = multiprocessing.cpu_count() # Count the number of cores

####################### MAKE OUTPUT DIRECTORY ############################
outdir = './outputs/task1'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
###################### READ THE DATA ################################
print(colored('Reading data..', 'green', attrs=['reverse', 'blink']))
candidate_passages = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None)
# Add columns
candidate_passages.columns = ['qid', 'pid', 'queries', 'passage']
test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None)
training_data = pd.read_csv('train_data.tsv', sep='\t',  low_memory=False) 
# first row s the header
validation_data = pd.read_csv('validation_data.tsv', sep='\t',  low_memory=False) 
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))

##################### SUBSAMPLE DATA: 15% TEST, 15% VAL, 70% TRAIN #######################
# since test data has 189877, approximately train should be 886093
training_data_sample = training_data.sample(n=886093, random_state=1) 
# validation data has 189877
validation_data_sample = validation_data.sample(n=189877, random_state=1)


############################## DROP DUPLICATES FOR FASTER TOKENIZATION ############################
validation_data_sample_unique = validation_data_sample.drop_duplicates(subset=['pid'])
validation_data_sample_unique_q = validation_data_sample.drop_duplicates(subset=['qid'])
training_data_sample_unique = training_data_sample.drop_duplicates(subset=['pid'])
training_data_sample_unique_q = training_data_sample.drop_duplicates(subset=['qid'])


################## NEED PIDS AND QIDS   ###############################################
validation_pid = validation_data_sample['pid'].unique()
candidate_pid= candidate_passages['pid'].unique().tolist()
test_qid = test_queries[0].unique().tolist() #in candidate passages
validation_qid = validation_data_sample['qid'].unique().tolist() # list of query ids (numbers)
t_pid = training_data_sample_unique['pid'].tolist()
t_qid = training_data_sample_unique_q['qid'].tolist()


#######################CLEAN TEXT ########################################
def cleaning(doc):
    # Lemmatizes and removes stopwords. returns list of lists containing each lemma as a different element 
    # doc needs to be a spacy Doc object
    txt =[token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 1]
    
    return ' '.join(txt).split() 

query_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in test_queries[1])
txt_q_test = [cleaning(doc) for doc in tqdm(nlp.pipe(query_cleaning, batch_size=3500, n_process = cores-1))]
path = outdir + '/test_queries_clean.json'
with open(path, 'w') as f:
    json.dump(txt_q_test, f)


candidate_unique = candidate_passages.drop_duplicates(subset=['pid'])
candidate_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in candidate_unique['passage'])
txt_candidate = [cleaning(doc) for doc in tqdm(nlp.pipe(candidate_cleaning, batch_size=3500, n_process = cores-1))]
    
path = outdir + '/candidate_clean.json'
with open(path, 'w') as f:
    json.dump(txt_candidate, f)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in training_data_sample_unique['passage'])
txt= [cleaning(doc) for doc in tqdm(nlp.pipe(brief_cleaning, batch_size=3500, n_process = cores-1))]
path = outdir + '/training_passages_clean.json'
with open(path, 'w') as f:
    json.dump(txt, f)

brief_cleaning_q = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in training_data_sample_unique_q['queries'])
txt_q = [cleaning(doc) for doc in tqdm(nlp.pipe(brief_cleaning_q, batch_size=4000, n_process = cores-1))]
path = outdir + '/training_queries_clean.json'
with open(path, 'w') as f:
    json.dump(txt_q, f)

val_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in validation_data_sample_unique['passage'])
txt_val = [cleaning(doc) for doc in tqdm(nlp.pipe(val_cleaning, batch_size=3000, n_process = cores-1))]
path = outdir + '/validation_passages_clean.json'
with open(path, 'w') as f:
    json.dump(txt_val, f)

val_cleaning_q = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in validation_data_sample_unique_q['queries'])
txt_q_val = [cleaning(doc) for doc in tqdm(nlp.pipe(val_cleaning_q, batch_size=4000, n_process = cores-1))]

path = outdir + '/validation_queries_clean.json'
with open(path, 'w') as f:
    json.dump(txt_q_val, f)

################## #associate text with ID
txt_q_dict = dict(zip(t_qid, txt_q))
txt_dict = dict(zip(t_pid, txt)) 
txt_q_val_dict = dict(zip(validation_qid, txt_q_val))
txt_val_dict = dict(zip(validation_pid, txt_val))
txt_q_test_dict = dict(zip(test_qid, txt_q_test))
txt_candidate_dict = dict(zip(candidate_pid, txt_candidate))

### add clean text to dataframes
training_data_sample['clean queries'] = training_data_sample['qid'].map(txt_q_dict)
training_data_sample['clean passages'] = training_data_sample['pid'].map(txt_dict)
validation_data_sample['clean queries'] = validation_data_sample['qid'].map(txt_q_val_dict)
validation_data_sample['clean passages'] = validation_data_sample['pid'].map(txt_val_dict)
candidate_passages['clean passages'] = candidate_passages['pid'].map(txt_candidate_dict)
candidate_passages['clean queries'] = candidate_passages['qid'].map(txt_q_test_dict)


############## SAVE  CLEANED DATA #####################################
training_data_sample[['qid','pid', 'clean queries', 'clean passages', 'relevancy' ]].to_csv(outdir + '/training_data_sample_clean.txt', sep=',', index=False) 
validation_data_sample[['qid','pid', 'clean queries', 'clean passages', 'relevancy' ]].to_csv(outdir + '/validation_data_sample_clean.txt', sep=',', index=False) 
candidate_passages[['qid','pid', 'clean queries', 'clean passages' ]].to_csv(outdir + '/candidate_passages_clean.txt', sep=',', index=False)
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

################# RANK VALIDATION DATA ###########################
validation_data_sample['bm25'] = bm25_feature(validation_data_sample, validation_qid)

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

################### EVALUATE BM25 ############################
validation_data_sample['rank'] = validation_data_sample.groupby('qid')['bm25'].rank(ascending=False, method='first')
validation_data_sample['rank'] = validation_data_sample['rank'].astype(int)
mean_precision_bm25, retrieved_validation_bm25 = mean_precision(validation_data_sample, validation_qid)
mean_ndcg_bm25 = mean_ndcg(validation_data_sample, retrieved_validation_bm25)
print('Mean Precision BM25: ', mean_precision_bm25)
print('Mean NDCG BM25: ', mean_ndcg_bm25)

# Save the results
with open(outdir + '/bm25_results.txt', 'w') as f:
    f.write('Mean Precision BM25: ' + str(mean_precision_bm25) + '\n')
    f.write('Mean NDCG BM25: ' + str(mean_ndcg_bm25) + '\n')

