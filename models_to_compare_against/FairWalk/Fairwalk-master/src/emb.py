#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:24:00 2018

@author: yangzhang, enderged
"""

import multiprocessing as mp
import networkx as nx

import pandas as pd
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from graph_utils import *

from gensim.models import KeyedVectors, word2vec



def pagerank_bias_random_walk(nxgraph, out_path, walk_len=100, walk_times=20):
    pr_scores = nx.pagerank(nxgraph).values()
    with open(out_path, 'w') as out:
        for start_node in nxgraph.nodes_iter():
            print("{:>4} / {:>4}".format(start_node, len(nxgraph)))
            for _ in xrange(walk_times):
                walk = np.zeros(walk_len, dtype=int)
                walk[0] = start_node
                for i in xrange(1, walk_len):
                    neigh = nxgraph.neighbors(walk[i-1])
                    weights = [pr_scores[x] for x in neigh]
                    sumw = sum(weights)
                    weights = [x / sumw for x in weights]
                    walk[i] = np.random.choice(neigh, 1, p=weights)
                out.write(",".join([str(x) for x in walk]))
                out.write("\n")



def emb_train(path, name, walk_len=80, walk_times=20, num_features=128):
    """
    run word2vec on a given walks file
    walks can be generated fast by Christian's program
    """
    walks = pd.read_csv('{}/{}.walk'.format(path, name), header=None)

    walks = walks.loc[np.random.permutation(len(walks))]
    walks = walks.reset_index(drop=True)
    walks = walks.applymap(str) # gensim only accept list of strings
    
    
    min_word_count = 10
    num_workers = mp.cpu_count()
    context = 10
    downsampling = 1e-3
    
    # gensim does not support numpy array, thus, walks.tolist()
    walks = walks.groupby(0).head(walk_times).values[:,:walk_len].tolist()
    emb = word2vec.Word2Vec(walks,\
                            sg=1,\
                            workers=num_workers,\
                            size=num_features, min_count=min_word_count,\
                            window=context, sample=downsampling)
    print 'training done'
    emb.wv.save_word2vec_format('{}/{}_{}_{}_{}.emb'.format(
            path, name,
            str(int(walk_len)),
            str(int(walk_times)),
            str(int(num_features))
            ))
    emb.wv.save('{}/{}_{}_{}_{}.model'.format(
            path, name,
            str(int(walk_len)),
            str(int(walk_times)),
            str(int(num_features))
            ))
    
def train_5_emb(path='.', name='known_80'):
    for i in range(5):
        new_path = '{}/{}'.format(path, str(i))
        emb_train(new_path, name)
    
    
    
def get_emb(path):
    with open(path, 'r') as ff:
        emb = ff.readlines()
        emb2 = sorted([[float(i) for i in vec.split()] for vec in emb[1:]])
        #emb3 = [i[1:] for i in emb2]
    return emb2


def get_most_similar(path, path2, postfix='_80_20_128', n=250):
    """
    Probably Deprecated
    
    From `path`.model read WordVectors and to each, find their `n` most similar
    (by cosine similarity) nodes. Save it in `path`_top`n`.json.
    Filter it by known file.
    """
    wv = KeyedVectors.load(path + path2 + postfix + '.model')
    known = read_instagram_known(path + '.known')
    nxgraph = read_nxgraph(path + path2 + '.edgelist')
    wv2 = KeyedVectors(vector_size = 128)
    # filter by known
    for ent, vec in zip(wv.index2entity, wv.vectors):
        if int(ent) in known:
            wv2.add([ent], [vec])
    samples = dict([
            (int(u), [
                    (int(i[0]), i[1])
                    for i in wv.most_similar([u], topn=n)
                    if int(i[0]) not in nxgraph.neighbors(int(u))
                    ])
            for u in wv2.vocab.iterkeys()
            ])

    write_pickle(samples, '{}_top{}.pick'.format(path, n))
    
    
def get_most_similar_from_5_known(path, postfix='gendeq_', n=100):
    """
    For each file from:
        `path`/{0,1,2,3,4}/known_80_`postfix`80_20_128.model
    calculate `n` most similar nodes that are not friends in:
        `path`/{0,1,2,3,4}/known_80.edgelist
    find whether they are real friends or not from:
        `path`/{0,1,2,3,4}/known_20.edgelist
    and save it to:
        `path`/{0,1,2,3,4}/known_80_`postfix`top`n`.pick
    """
    for i in range(5):
        print('starting {}'.format(str(i)))
        new_path = '{}/{}/known_'.format(path, str(i))
        wv = KeyedVectors.load('{}80_{}80_20_128.model'.format(new_path, postfix))
        nxgraph = read_nxgraph(new_path + '80.edgelist')
        true_graph = read_nxgraph(new_path + '20.edgelist')
        samples = dict([
            (int(u), [
                    (int(i[0]), int(int(u) in true_graph.nodes() and int(i[0]) in true_graph.neighbors(int(u))))
                    for i in wv.most_similar([u], topn=n+100)
                    if int(i[0]) not in nxgraph.neighbors(int(u))
                    ][:n])
            for u in wv.vocab.iterkeys()
            ])
        print('finishing {}'.format(str(i)))
        write_pickle(samples, '{}80_{}top{}.pick'.format(new_path, postfix, str(n)))
        
        
def get_counts(walk_file):
        walks = [int(i) for i in ','.join([line.strip() for line in open(walk_file, 'r').readlines()]).split(',')]
        return Counter(walks)
        