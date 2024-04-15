######## TASK 2 - LOGISTIC REGRESSION MODEL WITH ############
################## WORD2VEC EMBEDDINGS ###################################
######################################################################

# Importing the necessary libraries
import json
import pandas as pd
import numpy as np
from termcolor import colored
from tqdm import tqdm
import re  
import os
from collections import defaultdict 
import scipy as sp
from scipy import spatial
from scipy import stats
from scipy.optimize import fmin
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", 
                    datefmt= '%H:%M:%S', level=logging.INFO)
import multiprocessing
cores = multiprocessing.cpu_count() 
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
####################### MAKE OUTPUT DIRECTORY ############################
outdir = './outputs/task2'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dir_task1 = './outputs/task1'
############# LOADING THE CLEAN DATA FROM TASK 1############################
# same sampling as task 1 
training_data_sample = pd.read_csv(dir_task1 + '/training_data_sample_clean.txt', sep=',')
validation_data = pd.read_csv(dir_task1+ '/validation_data_sample_clean.txt', sep=',')
candidate_passages = pd.read_csv(dir_task1 + '/candidate_passages_clean.txt', sep=',')
############### LOAD CLEAN TEXT ################
# Clean text 
with open(dir_task1 + '/training_passages_clean.json', 'r') as f:
    txt = json.load(f)
with open(dir_task1 + '/training_queries_clean.json', 'r') as f:
    txt_q = json.load(f)
with open(dir_task1 + '/validation_queries_clean.json', 'r') as f:
    txt_q_val = json.load(f)
with open(dir_task1 + '/validation_passages_clean.json', 'r') as f:
    txt_val = json.load(f)
with open(dir_task1 + '/test_queries_clean.json', 'r') as f:
    txt_test = json.load(f)
with open(dir_task1 + '/candidate_clean.json', 'r') as f:
    txt_test_p = json.load(f)

#concatenate all the training text for word2vec model
full_doc_clean = txt + txt_q


training_data_sample_unique = training_data_sample.drop_duplicates(subset='pid')
validation_data_unique = validation_data.drop_duplicates(subset='pid')
candidate_passages_unique = candidate_passages.drop_duplicates(subset='pid')
training_data_sample_unique_q = training_data_sample.drop_duplicates(subset='qid')
validation_data_unique_q = validation_data.drop_duplicates(subset='qid')
candidate_passages_unique_q = candidate_passages.drop_duplicates(subset='qid')

#### PIDS AND QIDS ####
t_qid = training_data_sample_unique_q['qid'].values
t_pid = training_data_sample_unique['pid'].values
validation_pid = validation_data_unique['pid'].values
validation_qid = validation_data_unique_q['qid'].values
candidate_pid = candidate_passages_unique['pid'].values
test_qid = candidate_passages_unique_q['qid'].values



############### WORD2VEC MODEL ############################
class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1

loss_logger = LossLogger()

w2v_model = Word2Vec(min_count=5,
                     window=4,
                     vector_size=200,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=10,#negative sampling. 10 noise words
                     workers=cores,
                     callbacks=[loss_logger],
                    compute_loss=True)
w2v_model.build_vocab(full_doc_clean, progress_per=10000)
w2v_model.train(full_doc_clean, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1,  callbacks=[loss_logger],
                    compute_loss=True)

#### PRINT MODEL LOSSES AT EACH EPOCH ####
ls = loss_logger.losses
for index, i in enumerate(ls):
    if index > 0:
        print(i - ls[index-1])

# Store just the words + their trained embeddings for later use (less memory).
word_vectors = w2v_model.wv
word_vectors.save(outdir + "/word2vec.wordvectors")
# Load back with memory-mapping = read-only,
wv = KeyedVectors.load(outdir + "/word2vec.wordvectors", mmap='r')
# Get the vocabulary
vocabulary = wv.key_to_index

################################################################
######################### FUNCTIONS #############################################
########################################################################

######################FOR WORD AND DOCUMENT EMBEDDING ########################

def embeddings(word, model, rand = False):
    if word in vocabulary:
        return model.get_vector(word)
    else:
        if rand:
            return np.random.rand(200)
        else: # to drop in training so not to train poorly
            return np.full(200, np.nan) # return nan vector if word not in vocabulary and rand is False

def avg_embedding(doc, model, rand = False):
    embeddings_array = np.array([embeddings(x, model, rand) for x in doc])
    if np.all(np.isnan(embeddings_array)): # if all embeddings are nan, return nan
        return np.full(200, np.nan)
    else:
        return np.nanmean(embeddings_array, axis=0) # use np.nanmean to ignore nan values

def doc_embedding(list_docs, list_keys, model, rand= False ):
    store_dic = {}
    for index, sen in tqdm(enumerate(list_docs)):
        average_vector = avg_embedding(sen, model, rand)
        if np.isnan(average_vector).any(): # skip if average_vector is nan
            continue
        d1 = {list_keys[index]: (average_vector)}
        store_dic.update(d1)
    return store_dic

################  FOR COSINE SIMILARITY BASED ON EMBEDDINGS #######################
def get_sim(query_embedding, average_vec): #cosine similarity
    sim = (1 - sp.spatial.distance.cosine(query_embedding, average_vec))
    #Adjust the similarity to be between 0 and 1
    sim = (sim + 1) / 2
    return sim
# claculate cosine similarity based on embeddings
def calculate_similarity(row):
    query_embedding = row['query_embedding']
    passage_embedding = row['passage_embedding']
    if not isinstance(query_embedding, np.ndarray) or not isinstance(passage_embedding, np.ndarray):
        return np.nan #if no embeddings
    sim = get_sim(query_embedding, passage_embedding)
    return sim


############## FOR LOGISTIC REGRESSION MODEL ############################
# Define logistic function
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# Define the gradient descent algorithm with binary cross entropy loss
def mini_batch_gradient_descent(X, y, lr, weights, n_iter=10, batch_size=5, verbose=True):
    n_samples, n_features = X.shape
    bias = 0
    weights = np.array([weights])  # Ensure weights is a one-dimensional numpy array
    for i in tqdm(range(n_iter)):
        indices = np.random.permutation(n_samples)
        for j in range(0, n_samples, batch_size):
            batch_indices = indices[j:j+batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]

            model = np.dot(X_batch, weights) + bias
            predictions = sigmoid(model)

            # Gradient update
            dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
            db = (1 / len(X_batch)) * np.sum(predictions - y_batch)

            # Update weights and bias
            weights -= lr * dw
            bias -= lr * db

            # Loss calculation (binary cross-entropy)
            loss = -np.mean(y_batch * np.log(predictions + 1e-15) + (1 - y_batch) * np.log(1 - predictions + 1e-15))
        # Investigate rresults. Look at effect of learning rate and diffferent starting points of theta. See if they converge. 
        if verbose:
            print(f"Iteration {i}: Loss = {loss}, Learning Rate = {lr}")
    return weights, bias


def calculate_score(row):
    query_embedding = row['query_embedding']
    passage_embedding = row['passage_embedding']
    if not isinstance(query_embedding, np.ndarray) or not isinstance(passage_embedding, np.ndarray):
        return np.nan
    score = logistic_function(np.dot( weights, get_sim(query_embedding, passage_embedding)) + bias)
    return score

##################################################################
################## FUNCTIONS FOR PRECISION AND NDCG ###############
# Mean precision takes top 100 ranks for each query and calculates the precision   
def mean_precision(df, validation_qid):
    precision_per_qid = []
    retrieved = []
    for qid in validation_qid:
        if qid in df['qid'].unique():
            top100 = df[df['qid']==qid]['rank'].head(100).index
            overlap_count = df[(df['relevancy']==1) & (df.index.isin(top100))].shape[0]
            precision_per_qid.append(overlap_count/len(top100))
            retrieved.append(len(top100))
        else:
            print('Dropped a query because embedding was empty. QID:', qid)
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

###########################################################



################ CREATE EMBEDDINGS FOR TRAINING DATA ###############
# Average vector for each document. Keys are pids/qids
passage_vectors_train = doc_embedding(txt, t_pid, wv, rand = False)
query_vectors_train = doc_embedding(txt_q, t_qid, wv, rand = False)
# Map to the training data

training_data_sample['query_embedding'] = training_data_sample['qid'].map(query_vectors_train)
# drop rows with zero embeddings
training_data_sample.dropna(subset=['query_embedding'], inplace=True)
training_data_sample['passage_embedding'] = training_data_sample['pid'].map(passage_vectors_train)
training_data_sample.dropna(subset=['passage_embedding'], inplace=True)


########################### CALCULATE SIMILARITY BASED ON EMBEDDINGS ############################
training_data_sample['sim'] = training_data_sample.apply(calculate_similarity, axis=1)
# Drop the rows with np.nan in 'sim' column. Beware, now not the same rows as original data
training_data_sample.dropna(subset=['sim'], inplace=True)

# Create the X and y for the logistic regression model
y = training_data_sample['relevancy'].tolist() # class labels
X = training_data_sample['sim'].tolist() # cosine similarity scores
# Convert to numpy arrays for the model
X = np.array(X)
y = np.array(y)
X = X.reshape(-1, 1)

# investigate data
training_data_sample = training_data_sample.sort_values(by='sim', ascending= False)

training_data_sample.head(20) 

################## TRAINING THE LOGISTIC REGRESSION MODEL ############################
weights_in = np.linspace(-10, 10, 5)  # Linearly spaced weight values

results = []
weight_predictions = []
# Run gradient descent for each theta in the linspace array
for index, weight_initial in enumerate(weights_in):
    if index == 0:
        for lr in [0.1, 0.01,0.001]: # try different learning rates
            weight_pred, bias = mini_batch_stochastic_gradient_descent(
                X, y, lr, weights = weight_initial , n_iter=10, batch_size=5, verbose=True
            )
            results.append(f"alpha = {lr}, initial weight: {weight_initial}, weight_predicted: {weight_pred}, bias: {bias}")
            weight_predictions.append([weight_pred, bias])
    else:
         lr = 0.01
         weight_pred, bias = mini_batch_stochastic_gradient_descent(
             X, y, lr, weights = weight_initial , n_iter=10, batch_size=5, verbose=True
         )
         results.append(f"alpha = {lr}, initial weight: {weight_initial}, weight_predicted: {weight_pred}, bias: {bias}")
         weight_predictions.append([weight_pred, bias])
###################### SAVE THE MODEL TRAINING TO TEXT FILE ############################
# Save to text file
with open(outdir + '/model_training.txt', 'w') as f:
    for line in results:
        f.write(line + '\n')

########## Choose weight and bias to go forward ( call weights and bias which is fed into logistic scoring function),################
#investigate  and choose
print(weight_predictions)
weights = -4.92 
bias = -2.96

######################################################################################
################################### COMPARE WITH SCIKIT LEARN ########################################
####################################################################################
from sklearn.linear_model import LogisticRegression

# Fit the model
model = LogisticRegression()
model.fit(X, y)

# Generate a range of values for cosine similarity
sim_values = np.linspace(-5, 5, 500).reshape(-1, 1)

# Predict probabilities for these values
probs = model.predict_proba(sim_values)[:, 1]

# Plot the scatter plot
plt.scatter(training_data_sample['sim'], training_data_sample['relevancy'])
plt.xlabel('Cosine Similarity')
plt.ylabel('Relevancy')
plt.title('Relevancy vs Cosine Similarity')

# Add the logistic regression curve
plt.plot(sim_values, probs, color='red', label='Scikit-learn')
weights = weights.reshape(-1, 1)
prob_mine = logistic_function(np.dot(  sim_values, weights) + bias)
plt.plot(sim_values, prob_mine, color='green', label = 'Own implementation')
plt.legend(loc = 'center right')
plt.show()

#############################################################################
################################################################
########################### VALIDATION DATA ########################################
###################################################################
# Create embeddings for validation data
passage_vectors_v= doc_embedding(txt_val, validation_pid, wv, rand = True)
query_vectors_v = doc_embedding(txt_q_val, validation_qid, wv, rand = True)
# get similarity and probability of relevance (logistic score).
validation_data['query_embedding'] = validation_data['qid'].map(query_vectors_v)
# drop nan
validation_data.dropna(subset=['query_embedding'], inplace=True)
validation_data['passage_embedding'] = validation_data['pid'].map(passage_vectors_v)
validation_data.dropna(subset=['passage_embedding'], inplace=True)
validation_data['sim'] = validation_data.apply(calculate_similarity, axis=1)
validation_data.dropna(subset=['sim'], inplace=True)
validation_data['logistic_score'] = validation_data.apply(calculate_score, axis=1)

###################### EVALUATION METRICS ############################
# change rank to be based on logistic score
validation_data['rank'] = validation_data.groupby('qid')['logistic_score'].rank(ascending=False, method='first')
validation_data['rank'] = validation_data['rank'].astype(int)
# Calculate mean precision and mean NDCG
mean_precision_logi, retrieved_validation_logi = mean_precision(validation_data, validation_qid)
mean_ndcg_logi = mean_ndcg(validation_data, retrieved_validation_logi)

print('Mean Precision Logistic: ', mean_precision_logi)
print('Mean NDCG Logistic: ', mean_ndcg_logi)
with open(outdir + '/metrics_logi.txt', 'w') as f:
    f.write(f"Mean Precision Logistic: {mean_precision_logi}\n")
    f.write(f"Mean NDCG Logistic: {mean_ndcg_logi}\n")


####################### RE-RANKING TEST DATA ########################################
# Create embeddings for test data
# get similarity and probability of relevance
# make new column  called 'logistic_score'
passage_vectors_test= doc_embedding(txt_test_p, candidate_pid, wv, rand = True)
query_vectors_test = doc_embedding(txt_test, test_qid, wv, rand = True)
candidate_passages['query_embedding'] = candidate_passages['qid'].map(query_vectors_test)
candidate_passages.dropna(subset=['query_embedding'], inplace=True)
candidate_passages['passage_embedding'] = candidate_passages['pid'].map(passage_vectors_test)
candidate_passages.dropna(subset=['passage_embedding'], inplace=True)
candidate_passages['sim'] = candidate_passages.apply(calculate_similarity, axis=1)

candidate_passages.dropna(subset=['sim'], inplace=True)
candidate_passages['logistic_score'] = candidate_passages.apply(calculate_score, axis=1)
candidate_passages['rank'] = candidate_passages.groupby('qid')['logistic_score'].rank(ascending=False, method='first')
candidate_passages['rank'] = candidate_passages['rank'].astype(int)

# Need to retrieve only at most 100 documents per query
# Save the top 100 documents for each query
candidate_passages_copy = candidate_passages.copy()
top = pd.DataFrame()
test_qid = candidate_passages['qid'].unique() # Get the unique qids after dropping rows in case of NaN
# Use the copy in the loop
for qid in test_qid:
    top100 = candidate_passages_copy[candidate_passages_copy['qid'] == qid].sort_values(by='rank').head(100)
    top = pd.concat([top, top100], axis=0)
top['A2'] = 'A2'
top['algoname'] = 'LR'  # Logistic Regression
top = top[['qid', 'A2', 'pid', 'rank', 'logistic_score', 'algoname']]
# save to outdir
assert top.shape[0] == 10000
top.to_csv(outdir + '/LR.txt', sep=' ', index=False, header=False)


######################## SAVE DATA WITH SIMILARITY SCORES ############################
########################## FOR LAMBDA RANK ##########################################
# Beware !! not the same rows as original data because drop rows with np.nan in 'sim' column
training_data_sample[['qid','pid', 'clean queries', 'clean passages', 'relevancy', 'sim' ]].to_csv(outdir + '/training_data_sample_sim.txt', sep=',', index=False) 
validation_data[['qid','pid', 'clean queries', 'clean passages', 'relevancy','sim']].to_csv(outdir + '/validation_data_sim.txt', sep=',', index=False) 
candidate_passages[['qid','pid', 'clean queries', 'clean passages', 'sim' ]].to_csv(outdir + '/candidate_passages_sim.txt', sep=',', index=False)
