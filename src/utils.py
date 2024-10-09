
'''
***********************************************************************
CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- (I2) Category-Weighted High-Confidence Negative Sampling of CatDive
- (I3) Coverage-Prioritized Reranking of CatDive
- Dataset preperation including evaluation.

Version: 1.0
***********************************************************************
'''


import copy
import math
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from tqdm import tqdm
import itertools


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

  
'''
PROPOSED: Category-Weighted High-Confidence Negative Sampling of CatDive

input:
* beta: beta of Equation (7), controls the confidence of negative samples
* category: category of each item
* catenum: number of categories
* size: negative sample size
* ts: a set of items that the given user has interacted with
* itemnum: number of items
* popularity: popularity of each item
returns:
* neg_list: list of negative samples
'''
def catdive_ns(beta, category, catenum, size, ts, itemnum, popularity):
    
    score = np.array([0] * (itemnum+1), dtype=np.float32)
    
    category_pop = np.array(list(Counter(list(range(0, catenum+1)) + list(category[ts[:size]])).values()), dtype=np.float32)
    score = category_pop[category] # get number of interaction for each item's category
    score /= sum(score) # S_c(i)^cat
    score += beta * (popularity / sum(popularity)) # adds S_i^pop to final score
    
    score[ts] = 0
    score = score[1:]
    neg_list = random.choices(list(range(1, itemnum+1)), weights=score, k=size) # samples according to the final negative sampling score
    
    return neg_list


'''
Random Negative Sampling for original SASRec

input:
* size: negative sample size
* ts: a set of items that the given user has interacted with
* itemnum: number of items
returns:
* neg: negative samples
'''
def random_ns(itemnum, ts, size):
    neg = []
    count = 0
    while count < size:
        t = random.randint(1, itemnum)
        while t in ts:
            t = random.randint(1, itemnum)
        neg.append(t)
        count += 1
    return neg


'''
Sample function of all user sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* catenum: number of categories
* category: category of each item
* popularity: popularity of each item
* batch_size: size of batch
* maxlen: maximum length of user sequence
* result_queue: queue to save sampling result
* beta: beta of Category-Weighted High-Confidence Negative Sampling (see Equation (7))
returns:
* a train batch
'''
def sample_function(user_train, usernum, itemnum, catenum, category, popularity, batch_size, maxlen, result_queue, beta):
    def sample():

        user = np.random.randint(1, usernum+1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum+1)
        
        ts = user_train[user]
        
        if len(ts) <= maxlen:
            zeros = [0] * (maxlen-len(ts)+1)
            seq = ts[:-1]
            pos = ts[1:]
            size = len(pos)
            seq = zeros + seq
            pos = zeros + pos
            if beta != -1:
                neg = catdive_ns(beta, category, catenum, size, ts, itemnum, popularity)
            else:
                neg = random_ns(itemnum, ts, size)
            neg = zeros + neg
        else:
            if maxlen == 5:
                t = np.random.randint(maxlen, len(ts)-maxlen)
                seq = ts[t-maxlen:t]
                pos = ts[t-maxlen+1:t+1]
            else:
                seq = ts[-(maxlen+1):-1]
                pos = ts[-maxlen:]
            if beta != -1:
                neg = catdive_ns(beta, category, catenum, maxlen, ts, itemnum, popularity)
            else:
                neg = random_ns(itemnum, ts, maxlen)
                
        return (user, seq, pos, neg)
    
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


'''
Wrap Sampler to get all train sequences with negative samples

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* catenum: number of categories
* category: category of each item
* popularity: popularity of each item
* batch_size: size of batch
* maxlen: maximum length of user sequence
* n_workers: number of workers to use in sampling
* cd_neg: aplha to control I3. Adjusted negative sampling
returns:
* user train sequences with negative samples
'''
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, catenum, category, popularity, batch_size=64, maxlen=10, n_workers=10, beta=-1):
        self.result_queue = Queue(maxsize=n_workers * 20)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      catenum,
                                                      category,
                                                      popularity,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      beta
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


'''
Train and test data partition function

input:
* fname: file name of dataset
returns:
* train and test data with information of dataset
'''
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open('data/%s/ratings' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        user_train[user] = User[user][:-2]
        user_valid[user] = [User[user][-2]]
        user_test[user] = [User[user][-1]]
    
    # loads category and popularity information of items
    category = pd.read_csv('data/'+fname+'/category', header=None)
    category = np.concatenate(([0], category.to_numpy().flatten()))
    popularity = pd.read_csv('data/'+fname+'/popularity', header=None)
    popularity = np.concatenate(([0], popularity.to_numpy().flatten()))
    
    return [user_train, user_valid, user_test, usernum, itemnum, category.max(), category, popularity]


'''
Train and test data partition function

input:
* model: model to evaluate
* dataset: dataset to evaluate on
* args: model details
* test_f: true if test and false if validation
returns:
* HitRate
* nDCG
* Category Diversity
'''
def evaluate(model, dataset, args, test_f=False, lamb=0):
    [train, valid, test, usernum, itemnum, catenum, category, diversity] = copy.deepcopy(dataset)
    
    HT, ILD, nDCG, cov = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    if not test_f:
        users = random.sample(range(1, usernum+1), 1000)
    else:
        users = range(1, usernum+1)
        
    for u in tqdm(users):
        if test_f:
            train[u] = train[u] + valid[u]
        seq = train[u][-args.maxlen:]
        if len(seq) != args.maxlen:
            zeros = np.zeros(args.maxlen-len(seq), dtype=np.int32)
            seq = np.concatenate((zeros, seq), dtype=np.int32)

        item_idx = np.array(list(set(range(1, itemnum+1))-set(train[u])))
        predictions = model.predict(np.array([seq]), item_idx)
        
        if test_f:
            for i, k in enumerate([10, 20]):
                if lamb:
                    topk = catdive_rerank(predictions, item_idx, category, catenum, k, lamb)
                else:
                    _, topk = torch.topk(predictions, k)
                    topk = np.array(item_idx)[topk.cpu()]

                if test[u][0] in topk:
                    rank = np.where(topk == test[u][0])[0]
                    HT[i] += 1
                    nDCG[i] += 1.0 / np.log2(rank+2)
                
                comb = np.array(list(itertools.combinations(category[topk], 2))).T # gets combination of all items' category
                ILD[i] += len(np.where(comb[0]!=comb[1])[0]) / comb.shape[1]
                cov[i] += len(set(category[topk])) / catenum
        else:
            _, topk = torch.topk(predictions, args.topk)
            topk = np.array(item_idx)[topk.cpu()]
                        
            if valid[u][0] in topk:
                rank = np.where(topk == valid[u][0])[0]
                HT[0] += 1
                nDCG[0] += 1.0 / np.log2(rank+2)

            comb = np.array(list(itertools.combinations(category[topk], 2))).T
            ILD[0] += len(np.where(comb[0]!=comb[1])[0]) / comb.shape[1]
            cov[0] += len(set(category[topk])) / catenum
        
    return HT / len(users),  nDCG / len(users), ILD / len(users), cov / len(users)


'''
PROPOSED: Coverage-Prioritized Reranking

input:
* predictions: predicted score of items
* item_idx: index of items predicted
* category: category information of items
* catenum: number of categories
* k: number of items to recommend
* lamb: lambda of Equation (10), controls the diversity of recommendation
returns:
* top-k recommendation 
'''
def catdive_rerank(predictions, item_idx, category, catenum, k, lamb):
    score, index = torch.topk(predictions, 100)
    index = torch.tensor(item_idx[index.cpu()])
    unique_cat, counts = np.unique(category[index.cpu()], return_counts=True)
    top = []
    
    # selects items from the least popular categories of candidate items
    weights = np.zeros(catenum+1)
    unique_cat = unique_cat[np.argsort(counts)]
    for c in unique_cat[:int(k*lamb)]:
        choice = np.where(category[index.cpu()]==c)[0][0]
        top.append((index[choice], score[choice]))
        weights[category[choice]] += 1

    # rerank remaining candidate items according to the category distribution in current recommendation
    item_list = list(zip(index, score, category[index.cpu()]))
    cate_dict = defaultdict(int)
    while len(top) != k:
        max_index = 0
        max_score = item_list[0][1] - lamb * np.log(1+cate_dict[item_list[0][2]])
        for l in range(1, len(item_list)):
            if item_list[l][1] - lamb * np.log(1+cate_dict[item_list[l][2]]) > max_score:
                    max_index = l
                    max_score = item_list[l][1] - lamb * np.log(1+cate_dict[item_list[l][2]])
            elif item_list[l][1] < max_score:
                break
        if item_list[max_index][0] not in [x[0] for x in top]:
            top.append((item_list[max_index][0], item_list[max_index][1]))
            cate_dict[item_list[max_index][2]] += 1
        item_list.pop(max_index)
    
    # sort the items in the recommedation list according to their original score before reranking
    top = sorted(top, key=lambda x: x[1], reverse=True)
    return torch.stack([x[0] for x in top], dim=0).cpu()