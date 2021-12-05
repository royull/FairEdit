#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:33:01 2019

@author: enderged
"""

from fair import *
from emb import *

def main(data_path):

    ################################################################################
    # in Terminal, run bash scripts: walk.sh and walk_eq.sh using fast random walk project
    # https://github.com/EnderGed/fast-random-walk
    ################################################################################

    # get all 5 splits generate word2vec embeddings from walk files
    train_5_emb(data_path)
    train_5_emb(data_path, 'known_80_gendeq')
    train_5_emb(data_path, 'known_80_raceeq')

    # from embedding files get top most similar users for each user, for all 5 splits
    get_most_similar_from_5_known(data_path)
    get_most_similar_from_5_known(data_path, postfix='gendeq_')
    get_most_similar_from_5_known(data_path, postfix='raceeq_')

    # sample fake friends in amount equal to real friends for each node, for all 5 splits
    sample_false_edges_equal_amount_5_sets(data_path)

    # we perform friendship recommendation in supervised.main3.py


if __name__ == '__main__':
    data_path = '../data/la'
    main(data_path)
