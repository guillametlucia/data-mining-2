#################### TASK 4 - BERT#######################################
########################################################################

import json
import pandas as pd
import numpy as np
from termcolor import colored
from tqdm import tqdm
import os
from collections import defaultdict 
import scipy as sp
from scipy import spatial
from scipy import stats
from scipy.optimize import fmin
import multiprocessing
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
import torch
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding)
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

####################### MAKE OUTPUT DIRECTORY ############################
outdir = './outputs/task4'
if not os.path.exists(outdir):
    os.mkdir(outdir)

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

###################### SUBSAMPLE DATA ################################
# Separate the relevant and non-relevant instances
relevant = training_data[training_data['relevancy'] == 1]
non_relevant = training_data[training_data['relevancy'] == 0]
# Find the number to match
n_samples = min(len(relevant), len(non_relevant))
# Randomly sample from both groups same amount.
relevant_sample = relevant.sample(n=n_samples, random_state=1)
non_relevant_sample = non_relevant.sample(n=n_samples, random_state=1)
training_data_sample= pd.concat([relevant_sample, non_relevant_sample])

##################### SUBSAMPLE VALIDATION DATA #######################              
relevant_val = validation_data[validation_data['relevancy'] == 1]
non_relevant_val = validation_data[validation_data['relevancy'] == 0]
n_samples_val = min(len(relevant_val), len(non_relevant_val))
relevant_val_sample = relevant_val.sample(n=n_samples_val, random_state=1)
non_relevant_val_sample = non_relevant_val.sample(n=n_samples_val, random_state=1)
val_sample = pd.concat([relevant_val_sample, non_relevant_val_sample])
print('Training sample is now:', len(training_data_sample), 'rows long', ' Validation is:' , len(val_sample), 'rows long')
##################### PREPARE DATA #######################
train_bert = training_data_sample[['queries', 'passage', 'relevancy']]
val_bert_sample = val_sample[['queries', 'passage', 'relevancy']]

# Transform data to pytorch dataset
train_ds = Dataset.from_pandas(train_bert)
val_ds = Dataset.from_pandas(val_bert_sample)

# Load tokenizer from chosen model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Function to preprocess the data
def preprocess_data(examples):
    query = examples["queries"]
    passage = examples["passage"]
    encoding = tokenizer(query, passage, truncation= True, max_length  =512)
    encoding['labels'] = examples['relevancy']
    return encoding
# Tokenize and preprocess the data
encoded_train = train_ds.map(preprocess_data, batched=True)
encoded_train = encoded_train.remove_columns(['queries', 'passage', 'relevancy', '__index_level_0__'])
encoded_train.set_format('torch')

encoded_val = val_ds.map(preprocess_data, batched=True)
encoded_val = encoded_val.remove_columns(['queries', 'passage', 'relevancy', '__index_level_0__'])
encoded_val.set_format('torch')

# Data as dataloaders to be used in batches
# Collator for dynamic padding of input data
data_collator= DataCollatorWithPadding(tokenizer= tokenizer, return_tensors='pt')

val_dataloader = DataLoader(encoded_val, shuffle = False, batch_size = 151, collate_fn = data_collator  )


####### LOad model
BERTmodel = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                            num_labels=1,
                                                           use_cache= False)

############ TRAIN MODEL ############################

args = TrainingArguments(
    output_dir = outdir + '/bert_model',
    learning_rate=2e-5,
    save_strategy = 'epoch', #will save model to directory when epoch is done
    per_device_train_batch_size=13, #divisor of length of training data. Might have to adjust if memory problems.
    num_train_epochs=1, #only one epoch because of time
    weight_decay=0.01,
) 

trainer = Trainer(
    BERTmodel,
    args,
    train_dataset=encoded_train,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
trainer.save_model()

model_trained = AutoModelForSequenceClassification.from_pretrained( outdir + '/bert_model',
                                                          num_labels=1,
                                                            use_cache = False)

############# FUNCTIONS FOR EVALUATION ################
# Turn logits into probabilities.
def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict_probabilities(model_trained, data):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # if gpu is available, use it eg could run in colab for longer, more data.
    model_trained = model_trained.to(device)
    model_trained.eval()
    probabilities = []
    for batch in tqdm(data):
        batch = {k: v.to(device) for k, v in batch.items()}
        if 'labels' in batch:
            batch['labels'] = batch['labels'].long()  # Convert input_ids to Long
        with torch.no_grad():
            outputs = model_trained(**batch)
            logits = outputs.logits
            s = torch.sigmoid(logits)
            probabilities.extend(s.cpu().numpy()) 
    return probabilities

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

######################## EVALUATE ON VALIDATION DATA ############################
probabilities_val_sample = predict_probabilities(model_trained, val_dataloader)

val_sample['score'] = probabilities_val_sample
val_sample_qid = val_sample['qid'].unique()
val_sample['rank'] = val_sample.groupby('qid')['score'].rank(ascending=False, method='first')
val_sample['rank'] = val_sample['rank'].astype(int)
mean_precision_bert, retrieved_validation_bert = mean_precision(val_sample, val_sample_qid)
mean_ndcg_bert = mean_ndcg(val_sample, retrieved_validation_bert)

print(colored('Mean Precision BERT: ' + str(mean_precision_bert), 'green', attrs=['reverse', 'blink']))
print(colored('Mean NDCG BERT: ' + str(mean_ndcg_bert), 'green', attrs=['reverse', 'blink']))
with open(outdir + '/metrics_bert.txt', 'w') as f:
    f.write('MAP: ' + str(mean_precision_bert) + ' NDCG: ' + str(mean_ndcg_bert))

##############################################################
####################### RERANK TEST DATA #####################
##############################################################


############### PREPARE DATA ############################
def preprocess_candidate(examples):
    query = examples["queries"]
    passage = examples["passage"]
    encoding = tokenizer(query, passage, truncation= True, max_length  =512)
    return encoding

candidate_bert = candidate_passages[['queries', 'passage']]
# split to be able to batch the data without skipping any row.
candidate_ds_1 = Dataset.from_pandas(candidate_bert[:189824])
candidate_ds_2 = Dataset.from_pandas(candidate_bert[189824:])

encoded_candidate_1 = candidate_ds_1.map(preprocess_candidate, batched=True)
encoded_candidate_1 = encoded_candidate_1.remove_columns(['queries', 'passage'])
encoded_candidate_1.set_format('torch')
encoded_candidate_2 = candidate_ds_2.map(preprocess_candidate, batched=True)
encoded_candidate_2 = encoded_candidate_2.remove_columns(['queries', 'passage'])
encoded_candidate_2.set_format('torch')

candidate_dataloader_1 = DataLoader(encoded_candidate_1, shuffle = False, batch_size = 128, collate_fn = data_collator  )
candidate_dataloader_2 = DataLoader(encoded_candidate_2, shuffle = False, batch_size = 1, collate_fn = data_collator  )

###################### PREDICT PROBABILITIES ############################
probabilities_can1 = predict_probabilities(model_trained, candidate_dataloader_1)
probabilities_can2 = predict_probabilities(model_trained, candidate_dataloader_2)

#################### RERANKS ########################################
candidate_passages['bert_score'] = np.concatenate([probabilities_can1, probabilities_can2])
candidate_passages['rank'] = candidate_passages.groupby('qid')['bert_score'].rank(ascending=False, method='first')
candidate_passages['rank'] = candidate_passages['rank'].astype(int)
################# SAVE RERANKED DATA ################################
test_qid = candidate_passages['qid'].unique()
top = pd.DataFrame()
for qid in test_qid:
    top100 = candidate_passages[candidate_passages['qid'] == qid].sort_values(by='rank').head(100)
    top = pd.concat([top, top100], axis=0)
top['A2'] = 'A2'
top['algoname'] = 'NN' 
top = top[['qid', 'A2', 'pid', 'rank', 'bert_score', 'algoname']]
top.to_csv(outdir + '/NN.txt', sep=' ', index=False, header=False)
