
# Precision and recall from paragraph 6.5 of the paper

import sys
sys.path.append("../../src/")
import pandas as pd
from graph_utils import read_nxgraph
import networkx as nx
karr=[10000, 50000, 100000, 250000, 500000, 750000, 1000000]


for DATAPATH2 in ["../../data/la/", "../../data/london/"]:
    print DATAPATH2
    for toprecofile in ["topRecos.edgelist", "topRecos_geneq.edgelist", "topRecos_raceeq.edgelist"]:
        print toprecofile
        sum_prec, sum_recall = 0.0, 0.0
        prec_dict = {k: [] for k in karr}
        rec_dict = {k: [] for k in karr}
        for i in range(5):
            print "Iteration ", i
            DATAPATH = DATAPATH2 + str(i) + "/"

            G=read_nxgraph(DATAPATH+"known_20.edgelist")
            #relevant_df=nx.to_pandas_edgelist(G, source='u', target='v')
            #relevant_df=relevant_df.apply(sorted, axis=1)
            #relevant_df.drop_duplicates().to_csv(DATAPATH+"_sorted_nodups_known_20.csv", index=False)
            relevant_df=pd.read_csv(DATAPATH+"_sorted_nodups_known_20.csv")#, names=['u','v'] )

            reco_df=pd.read_csv(DATAPATH+toprecofile)
            reco_pairs=reco_df[['u', 'v']]
            #print "len(reco_pairs) may contain symmetric dups", len(reco_pairs)
            reco_pairs=reco_pairs.apply(sorted, axis=1).drop_duplicates()
            #print "after dropping dups", len(reco_pairs)
            reco_pairs.to_csv(DATAPATH+ "nodups_"+toprecofile, index=False)
            reco_pairs = pd.read_csv(DATAPATH+ "nodups_"+toprecofile)

            intersect = relevant_df.merge(reco_pairs)
            recall = float(len(intersect)) / float(len(relevant_df))



            precision = float(len(intersect) )/ float(len(reco_pairs))

            #print "full set len(intersect)", len(intersect), "recall, precision" , recall, precision

            sum_prec += precision
            sum_recall += recall

            for k in karr:
                reco_k=reco_pairs.iloc[-k:, :]
                intersect= relevant_df.merge(reco_k)
                #print "len(reco_k)", len(reco_k), "len(intersect)", len(intersect)

                recall_k= float(len(intersect))/float(len(relevant_df))
                precision_k= float(len(intersect))/float(len(reco_pairs))
                #print "recall, precision", recall, precision

                prec_dict[k].append(precision_k)
                rec_dict[k].append(recall_k)

                print prec_dict
                print rec_dict

        print "precision"
        for k in karr:
            print pd.np.average(prec_dict[k]) , ","
        print "all", float(sum_prec) / float(5)

        print "recall"
        for k in karr:
            print  pd.np.average(rec_dict[k]) , ","
        print "all", float(sum_recall) / float(5)