import sys
sys.path.append("../")
from pip._vendor import lockfile

from numpy.random import choice

from graph_utils import read_json, sample_false_edges, read_pickle, read_instagram_genrace

import pandas as pd
import networkx as nx

import multiprocessing as mp
from joblib import Parallel, delayed
#from bias_metrics import trainRF


def HADAMARD(u1, u2):
    return pd.np.multiply(u1, u2).values



def hada_pairs(filename,DATAPATH, pair, label):
    # if os.path.exists(filename):
    # os.remove(filename)

    emb = pd.read_csv(DATAPATH + 'Amherst41_100_80_128.emb', header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # last column is user id
    #emb = emb.loc[emb.uid > 0]  # only take users, no loc_type, not necessary

    count=0

    for u1, u2 in pair:

        u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])]
        u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])]

        try:
            val=HADAMARD(u1_vector, u2_vector)

            i_feature = pd.DataFrame([[u1, u2, label]])

            for i in range(0, emb.shape[1]-1):
                i_feature[i+3]=val[0][i]

            i_feature.to_csv(filename,\
                             index = False, header = None, mode = 'a')

        except Exception:
            print u1, u2
            count +=1
    print count , " pairs failed"



def hada_pairs_core(filename, DATAPATH, pair, label):
    # if os.path.exists(filename):
    # os.remove(filename)

    emb = pd.read_csv(DATAPATH + 'known_80_80_20_128.emb', header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # last column is user id
    #emb = emb.loc[emb.uid > 0]  # only take users, no loc_type, not necessary

    count1, count=0, 0
    outer_arr=[]
    for u1, u2 in pair:

        u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])]
        u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])]

        try:
            val=HADAMARD(u1_vector, u2_vector)
            in_arr = [u1, u2, label]

            for i in range(0, emb.shape[1]-1):
                in_arr.append(val[0][i])

            outer_arr.append(in_arr)
            if len(outer_arr)==1000:
                df = pd.DataFrame(outer_arr)
                df.to_csv(filename, index = False, header = None, mode = 'a')
                count1+=1
                print count1*1000, " rows"
                outer_arr = []

        except Exception:
            print u1, u2
            count +=1
    if len(outer_arr)!=0:
        df = pd.DataFrame(outer_arr)
        df.to_csv(filename, index=False, header=None, mode='a')
        print len(outer_arr), "rows"
    print count , " pairs failed"

def hada_pairs_core_threadsafe(emb, gender, filename, pair, label):

    count1, count=0, 0
    outer_arr=[]
    for u1, u2 in pair:

        u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])]
        u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])]

        try:
            val=HADAMARD(u1_vector, u2_vector)

            if gender[u1]==0:
                if gender[u2]==0:
                    group=0
                elif gender[u2]==1:
                   group=1
            elif gender[u1]==1:
                if gender[u2]==0:
                    group=2
                elif gender[u2]==1:
                    group=3

            in_arr = [u1, u2, label, group]

            for i in range(0, emb.shape[1]-1):
                in_arr.append(val[0][i])

            outer_arr.append(in_arr)


        except Exception:
            print u1, u2
            count +=1
    df = pd.DataFrame(outer_arr)
    with lockfile.LockFile(filename):
        df.to_csv(filename, index=False, header=None, mode='a')
    print len(outer_arr), "rows"
    print count , " pairs failed"



def hada_pairs_parallel(emb, gender, filename, pairs, label):

    core_num = mp.cpu_count()
    print "core_num", core_num

    sl_len=len(pairs)/core_num+1

    list_of_slices=zip(*(iter(pairs),) * sl_len)
    list_of_slices.append(tuple(pairs[sl_len * (core_num - 1):]))
    print "len(pairs), len(list_of_slices), slice length", len(pairs), len(list_of_slices), sl_len

    # do not use shared memory
    Parallel(n_jobs=core_num)(delayed(hada_pairs_core_threadsafe)(emb, gender, filename, slice, label) for slice in list_of_slices)

    return len(list_of_slices)* sl_len


def make_5_trainfiles_gen_eq(gender, DATAPATH2):
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"

        print "DATAPATH", DATAPATH
        nxgraph = nx.read_edgelist(DATAPATH + 'known_80.edgelist', delimiter=',', nodetype=int)
        print "edge list read"

        emb = pd.read_csv(DATAPATH + 'known_80_gendeq_80_20_128.emb', header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0: 'uid'})

        frn_count = hada_pairs_parallel(emb, gender, DATAPATH + "hada_tr_gendeq.csv", list(nxgraph.edges()),
                                        label=1)
        print "hada calculated for ", frn_count, " friend pairs"

        # we have 155100 users and 881657 frns, so we sample 6 strangers per user to get 930600

        uids = list(emb.uid.values)
        str_per_u = (frn_count / len(uids)) + 1

        dict_false = sample_false_edges(nxgraph, str_per_u)  # , uids)
        strangers = [(u, v) for u in dict_false.keys() for v in dict_false[u]]
        print len(strangers), " strangers sampled"

        str_count = hada_pairs_parallel(emb, gender, DATAPATH + 'hada_tr_gendeq.csv', strangers[:frn_count],
                                        label=0)
        print "hada calculated for ", str_count, " stranger pairs"




def make_5_recofiles(gender, DATAPATH2, embfile, pickfile_frn, pickfile_str, testfile_suffix):
    """
    for each split, calculate hadamard distance between candidate pairs
    :param gender: dictionary of uid, gender
    :param DATAPATH2:
    :param embfile: file containing embeddings of all users
    :param pickfile_frn: friend pairs
    :param pickfile_str: sampled stranger pairs
    :param testfile_suffix:
    :return:
    """

    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"

        emb = pd.read_csv(DATAPATH + embfile, header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0: 'uid'})

        top_dict = read_pickle(DATAPATH + pickfile_str)
        test_pairs_0 = [(u, v) for u in top_dict.keys() for v in top_dict[u]]
        print len(top_dict), len(test_pairs_0)
        count = hada_pairs_parallel(emb, gender, DATAPATH + testfile_suffix, test_pairs_0, label=0)
        print "hada calculated for ", count, " stranger pairs for candidate SET"

        top_dict = read_pickle(DATAPATH + pickfile_frn)
        test_pairs_1 = [ (u, v) for u in top_dict.keys() for [v, l] in top_dict[u] ]
        count = hada_pairs_parallel(emb, gender, DATAPATH + testfile_suffix, test_pairs_1, label=1)
        print "hada calculated for ", count, " friend pairs for candidate SET"


def make_5_testfiles(gender, DATAPATH2,embfile, edgelist, pickfile_str, testfile_suffix):

    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"

        emb = pd.read_csv(DATAPATH + embfile, header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0: 'uid'})


        nxgraph = nx.read_edgelist(DATAPATH + edgelist, delimiter=',', nodetype=int)
        print edgelist , "edge list read"
        count = hada_pairs_parallel(emb, gender, DATAPATH + testfile_suffix, list(nxgraph.edges()), label=1)
        print "hada calculated for ", count, " friend pairs for TEST SET"

        """top_dict = read_pickle(DATAPATH + pickfile_str)
        test_pairs_0 = [(u, v) for u in top_dict.keys() for v in top_dict[u]]
        print len(top_dict), len(test_pairs_0)
        count = hada_pairs_parallel(emb, gender, DATAPATH + testfile_suffix, test_pairs_0[:count], label=0)
        print "hada calculated for ", count, " stranger pairs for TEST SET"
        """






def make_5_trainfiles(gender, DATAPATH2):
    """
    reads true friends and samples stranger pairs
    and then calculates hadamard distances for training pairs
    :param gender:
    :param DATAPATH2:
    :return:
    """
    for i in range(5):

        DATAPATH = DATAPATH2 + str(i) + "/"

        print "DATAPATH", DATAPATH
        nxgraph = nx.read_edgelist(DATAPATH + 'known_80.edgelist', delimiter=',', nodetype=int)
        print "edge list read"
        emb = pd.read_csv(DATAPATH + 'known_80_80_20_128.emb', header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0: 'uid'})

        frn_count = hada_pairs_parallel(emb, gender, DATAPATH + "hada_tr.csv",  list(nxgraph.edges()), label=1)
        print "hada calculated for ", frn_count, " friend pairs"

        # we have 155100 users and 881657 frns, so we sample 6 strangers per user to get 930600

        uids = list(emb.uid.values)
        str_per_u = (frn_count / len(uids)) + 1

        dict_false = sample_false_edges(nxgraph, str_per_u)  # , uids)
        strangers = [(u, v) for u in dict_false.keys() for v in dict_false[u]]
        print len(strangers), " strangers sampled"

        str_count = hada_pairs_parallel(emb, gender, DATAPATH + 'hada_tr.csv',  strangers[:frn_count], label=0)
        print "hada calculated for ", str_count, " stranger pairs"









def make_training_insta(DATAPATH):

    nxgraph = nx.read_edgelist(DATAPATH+'london80.edgelist', delimiter=',', nodetype=int)
    print "edge list read"


    hada_pairs_parallel( DATAPATH+"hadafrns.csv", DATAPATH, list(nxgraph.edges()), label=1)

    # we have 155100 users and 881657 frns, so we sample 6 strangers per user to get 930600
    emb = pd.read_csv(DATAPATH + 'known_80_80_20_128.emb', header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})
    uids=list(emb.uid.values)

    dict_false=sample_false_edges(nxgraph, 6, uids)
    print  "read false edges"

    strangers= [ (u,v)  for u in dict_false.keys() for v in dict_false[u]]
    print len(strangers) , " strangers sampled"

    hada_pairs_parallel(DATAPATH+"hada3.csv", DATAPATH, strangers[:881657], label=0)


def test_data_pick(DATAPATH):

    top_dict = read_pickle(DATAPATH + "london_top250.pick")

    test_pairs = [(u, v) for u in top_dict.keys() for [v,c] in top_dict[u]]
    #:=LABELS
    #hada_pairs_parallel(DATAPATH + "hada_250_test.csv", DATAPATH, test_pairs, label=-1)


def get_top100(DATAPATH):
    #test_df = pd.read_csv('hada_100_test.csv', header=None, error_bad_lines=False)
    test_df = pd.read_csv(DATAPATH + 'hada_250_test.csv', header=None, error_bad_lines=False)  # , index_col=0)
    print test_df.shape
    test_df.dropna(inplace=True)
    print test_df.shape, " after dropna"
    df = pd.DataFrame()
    df['u'] = test_df.iloc[:, 0].values
    df['v'] = test_df.iloc[:, 1].values

    X_test=test_df.iloc[:, 3:].values

    clf=trainRF(DATAPATH)
    print "trained"
    pred_proba = clf.predict_proba(X_test)
    print len(pred_proba), " prob values obtained"

    df['prob'] = pred_proba[:, 1]
    #save the prob values somewhere
    top = df.groupby(['u']).apply(lambda x: x.nlargest(100, 'prob'))[['v', 'prob']]
    top.to_csv(DATAPATH+"supervised_250.csv")

    top=pd.read_csv(DATAPATH+"supervised_250.csv", usecols=['u', 'v', 'prob'])

    top.to_csv(DATAPATH+"supervised_250.csv")











def hada_pairs_chunky(filename,DATAPATH, pair, label):
    # if os.path.exists(filename):
    # os.remove(filename)

    emb = pd.read_csv(DATAPATH + 'Amherst41_100_80_128.emb', header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # last column is user id
    #emb = emb.loc[emb.uid > 0]  # only take users, no loc_type, not necessary

    count1, count=0, 0
    outer_arr=[]
    for u1, u2 in pair:

        u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])]
        u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])]

        try:
            val=HADAMARD(u1_vector, u2_vector)
            in_arr = [u1, u2, label]

            for i in range(0, emb.shape[1]-1):
                in_arr.append(val[0][i])

            outer_arr.append(in_arr)
            if len(outer_arr)==1000:
                df = pd.DataFrame(outer_arr)
                df.to_csv(filename, index = False, header = None, mode = 'a')
                count1+=1
                print count1*1000, " rows"
                outer_arr = []

        except Exception:

            print u1, u2
            count +=1
    print count , " pairs failed"




def make_training_data(DATAPATH):
    nxgraph = nx.read_edgelist(DATAPATH + 'Amherst41.edgelist', delimiter=',', nodetype=int)

    hada_pairs_parallel(DATAPATH+"hada.csv", DATAPATH, list(nxgraph.edges()), label=1)

    # we have 2235 users and 90954 frns, so we sample 41 strangers per user to get 91635
    arr_arr=sample_false_edges(nxgraph, 41)
    strangers= [ (u,v)  for u in range(len(arr_arr)) for v in arr_arr[u]]

    hada_pairs_parallel(DATAPATH+"hada.csv", DATAPATH, strangers[:90954], label=0)




def test_data(DATAPATH):

    test_arr_arr = read_json(DATAPATH + "sample1000.json")
    test_pairs = [(u, v) for u in test_arr_arr.keys() for [v,conf] in test_arr_arr[u]]

    hada_pairs_chunky(DATAPATH + "hada_1000_test.csv", DATAPATH, test_pairs, label=-1)


