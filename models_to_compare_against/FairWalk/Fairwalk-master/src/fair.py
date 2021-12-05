#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 09:21:27 2018

@author: enderged
"""

from graph_utils import *

import numpy as np


def equal_walk_prep(graph_in, graph_out, attr, no_attr):
    """
    Prepare graph at `graph_in` for random walk such that for each
    friend, the probability to going to any group of attributes (considerring
    `no_attr` number of groups) is equal.
    `attr` is a dictionary (not a path)
    Save the results in `graph_out`_`attr_name`eq.edgelist`
    """
    graph = read_nxgraph(graph_in)
    with open(graph_out, 'w') as out:
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            attrs = [attr[n] for n in neighbors]
            attr_count = [attrs.count(i) for i in range(no_attr)]
            # non zero product
            product = np.prod([i for i in attr_count if i != 0])
            attr_weight = [product / cnt if cnt != 0 else 0 for cnt in attr_count]
            for neigh in neighbors:
                out.write('{},{},{}\n'.format(node, neigh, attr_weight[attr[neigh]]))
    

def prep_5_equal_walks_insta(path, name):
    """
    Prepare 5 race-fair and 5 gender-fair weighted edgelist files
    Those files can be directly fed into random_walk.
    """
    gender, race = read_instagram_genrace('{}/{}.genrace'.format(path, name))
    for i in range(5):
        new_path = '{}/{}/known_80'.format(path, str(i))
        equal_walk_prep(
                new_path + '.edgelist',
                new_path + '_gendeq.edgelist',
                gender,
                2
                )
        equal_walk_prep(
                new_path + '.edgelist',
                new_path + '_raceeq.edgelist',
                race,
                3
                )
    with open('{}/walks_eq.sh'.format(path), 'w') as f:
        for i in range(5):
            f.write('~/random_walk/walk --if=./{}/known_80_gendeq.edgelist --of=./{}/known_80_gendeq.walk --length=80 --walks=20 -w\n'.format(i, i)) 
            f.write('~/random_walk/walk --if=./{}/known_80_raceeq.edgelist --of=./{}/known_80_raceeq.walk --length=80 --walks=20 -w\n'.format(i, i))            
        

    