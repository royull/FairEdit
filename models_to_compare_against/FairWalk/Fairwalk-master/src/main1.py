#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:33:01 2019

@author: enderged
"""

from fair import *
from emb import *

def main(data_path, data_name):
    prepare_instagram_data('{}/{}'.format(data_path, data_name))
    # remove all people with unknown or uncertain attributes, save result in _known file
    filter_graph_known_only('{}/{}'.format(data_path, data_name))
    # prepare training and testing data and prepares walks.sh
    split_graph_in_5(data_path, '{}_known.edgelist'.format(data_name))
    # prepare fairly weighted graph for random walks, generates walks_eq.sh
    prep_5_equal_walks_insta(data_path, data_name)

    ################################################################################
    # Next in Terminal, run bash scripts: walk.sh and walk_eq.sh using fast random walk project:
    # https://github.com/EnderGed/fast-random-walk
    ################################################################################




if __name__ == '__main__':
    data_path = '../data/la'
    data_name = 'la'
    main(data_path, data_name)
