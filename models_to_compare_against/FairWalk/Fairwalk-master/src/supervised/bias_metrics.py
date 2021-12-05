import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
sys.path.append("../../src/")
from graph_utils import read_instagram_genrace, read_json, read_nxgraph, write_nxgraph
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import networkx as nx


def trainRF(tr_filepath):

    train_pairs = pd.read_csv(tr_filepath, header=None)#, index_col=0)
    X_train, y_train = train_pairs.iloc[:, 4:].values, train_pairs.iloc[:, 2].values
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-2)
    clf.fit(X_train, y_train)

    return clf

def group_race(row, race):
    #group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8

    if (race[row['u1']] == 0) & (race[row['u2']] == 0): return  0
    if (race[row['u1']] == 0) & (race[row['u2']] == 1): return  1
    if (race[row['u1']] == 1) & (race[row['u2']] == 0): return  2
    if (race[row['u1']] == 1) & ( race[row['u2']]==1): return 3
    if (race[row['u1']] == 0) & (race[row['u2']] == 2): return  4
    if (race[row['u1']] == 1) & (race[row['u2']] == 2): return  5
    if (race[row['u1']] == 2) & (race[row['u2']] == 0): return  6
    if (race[row['u1']] == 2) & ( race[row['u2']]==1): return 7
    if (race[row['u1']] == 2) & ( race[row['u2']]==2): return 8


def getROCs_race(race, DATAPATH2, rocfile_suffix,testfile_suffix):

    colsNames=['u1', 'u2', 'label', 'group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    ROC_dict=dict.fromkeys(range(9), pd.DataFrame(columns=['th', 'fpr', 'tpr']))
    aucar=[]
    grp_auc_dict={k: [] for k in range(9)}

    for i in range(5):
        print "Iteration ", i
        DATAPATH=DATAPATH2+str(i)+"/"

        clf=trainRF(DATAPATH +'hada_tr.csv')
        #print "RF trained"

        test_df = pd.read_csv(DATAPATH + testfile_suffix, header=None, error_bad_lines=False, names=colsNames)  # , index_col=0)
        print test_df.shape, "test_df.shape"
        test_df.dropna(inplace=True)

        X_test_ = test_df.iloc[:, 4:].values
        y_proba_ = clf.predict_proba(X_test_)
        aucc= roc_auc_score(test_df.label.values.tolist(), y_proba_[:, 1])
        aucar.append(aucc)
        print aucc

        test_df['race_grp'] = test_df.apply(group_race, axis=1, args=(race,))
        test_df.to_csv(DATAPATH+'race_groups.csv')


        #group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8

        for gid in range(9):
            X_test= test_df[test_df.race_grp==gid].iloc[:,4:-1].values
            y_proba = clf.predict_proba(X_test)
            y_true = test_df[test_df.race_grp==gid].label.values.tolist()
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba[:, 1], pos_label=1)
            df_i_gid = pd.DataFrame({'th':thresholds, 'fpr':fpr,'tpr': tpr})
            print df_i_gid.shape
            ROC_dict[gid]=ROC_dict[gid].append(df_i_gid, ignore_index=True)
            grp_auc=roc_auc_score(y_true, y_proba[:, 1])
            print "gid, roc_auc_score(y_true, y_proba)", gid, grp_auc
            grp_auc_dict[gid].append(grp_auc)
            #print ROC_dict[gid].head(3)
            #print ROC_dict[gid].shape


    print "average AUC across 5 iterations for each group: "
    print "gid, AUC"
    for gid in range(9):
        ROC_dict[gid].to_csv(DATAPATH2 + str(gid) + rocfile_suffix)
        print gid, pd.np.average(grp_auc_dict[gid])

    print "avg auc of all iterations and groups", pd.np.average(aucar)
    return ROC_dict



# def get_Equality_opportunity(DATAPATH2, testfile_suffix, trainfile_suffix):
#
#     colsNames = ['u1', 'u2', 'label', 'group']
#     for i in range(0, 128):
#         name = "comp_" + str(i)
#         colsNames.append(name)
#
#     ARdict = dict.fromkeys(range(4))
#
#     AR_arr_dict = {k: [] for k in range(4)}
#     for i in range(5):
#         print "Iteration ", i
#         DATAPATH = DATAPATH2 + str(i) + "/"
#
#         clf = trainRF(DATAPATH + 'hada_tr.csv')
#         # print "RF trained"
#
#         test_df = pd.read_csv(DATAPATH + testfile_suffix, header=None, error_bad_lines=False,
#                               names=colsNames)  # , index_col=0)
#         print test_df.shape, "test_df.shape"
#         test_df.dropna(inplace=True)
#         # print test_df.shape, "test_df.shape after dropna"
#
#         # group: 00--0, 01--1, 10--2, 11--3
#         for gid in [0, 1, 2, 3]:
#             X_test = test_df[test_df.group == gid].iloc[:, 4:].values
#             y_true = test_df[test_df.group == gid].label.values.tolist()
#
#             y_pred = clf.predict(X_test)  # ,y_test)
#             tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#             print "group, tn, fp, fn, tp", gid, tn, fp, fn, tp
#             AR_arr_dict[gid].append(tp + fp)
#
#     print "avg AR across 5 iterations for each group: "
#     print "gid, AR"
#
#     for gid in [0,1,2,3]:
#         av[gid]=pd.np.average(AR_arr_dict[gid])
#         print gid, pd.np.average(AR_arr_dict[gid])
#         D_01, D_02, D_03=
#
#     return ROC_dict


def getROCs(DATAPATH2, rocfile_suffix,testfile_suffix):

    colsNames=['u1', 'u2', 'label', 'group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    ROC_dict=dict.fromkeys([0,1,2,3], pd.DataFrame(columns=['th', 'fpr', 'tpr']))
    aucar=[]
    grp_auc_dict={k: [] for k in range(4)}
    AR_arr_dict = {k: [] for k in range(4)}
    for i in range(5):
        print "Iteration ", i
        DATAPATH=DATAPATH2+str(i)+"/"

        clf=trainRF(DATAPATH +'hada_tr.csv')
        #print "RF trained"

        test_df = pd.read_csv(DATAPATH + testfile_suffix, header=None, error_bad_lines=False, names=colsNames)  # , index_col=0)
        print test_df.shape, "test_df.shape"
        test_df.dropna(inplace=True)
        #print test_df.shape, "test_df.shape after dropna"

        X_test_ = test_df.iloc[:, 4:].values
        y_proba_ = clf.predict_proba(X_test_)
        aucc= roc_auc_score(test_df.label.values.tolist(), y_proba_[:, 1])
        aucar.append(aucc)
        print aucc

        #group: 00--0, 01--1, 10--2, 11--3
        for gid in [0,1,2,3]:

            X_test= test_df[test_df.group==gid].iloc[:,4:].values
            y_true = test_df[test_df.group==gid].label.values.tolist()


            y_pred = clf.predict(X_test)   #,y_test)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print "group, tn, fp, fn, tp", gid, tn, fp, fn, tp
            AR_arr_dict[gid].append(tp+fp)



            y_proba = clf.predict_proba(X_test)   #,y_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba[:, 1], pos_label=1)
            df_i_gid = pd.DataFrame({'th':thresholds, 'fpr':fpr,'tpr': tpr})
            print df_i_gid.shape
            ROC_dict[gid]=ROC_dict[gid].append(df_i_gid, ignore_index=True)

            grp_auc=roc_auc_score(y_true, y_proba[:, 1])
            print "gid, roc_auc_score(y_true, y_proba)", gid, grp_auc
            grp_auc_dict[gid].append(grp_auc)
            #print  grp_auc_dict.items()

    print "average AUC across 5 iterations for each group: "
    print "gid, AUC"
    for gid in [0,1,2,3]:
        ROC_dict[gid].to_csv(DATAPATH2+str(gid)+ rocfile_suffix)
        print gid, pd.np.average(grp_auc_dict[gid])

    print "avg auc of all iterations and groups", pd.np.average(aucar)

    return ROC_dict




def sort_Reco(DATAPATH2, testfile, reco_file):
    """
    trains a classifier that predicts how likely a pair would be friends
    save recommendations sorted by the likelihood of being friends
    :param DATAPATH2:
    :param testfile:candidate pairs with hadamard distances
    :param reco_file:file containing indices pointing to top recommended pairs in testfile, along with the recommendation score
    :return:
    """

    for i in range(5):
        print "Iteration ", i
        DATAPATH=DATAPATH2+str(i)+"/"

        clf=trainRF(DATAPATH +'hada_tr.csv')
        print "RF trained"

        test_df = pd.read_csv(DATAPATH +testfile, header=None, error_bad_lines=False)#, names=colsNames)  # , index_col=0)

        X_test = test_df.iloc[:, 4:].values
        y_proba = clf.predict_proba(X_test)[:, 1]

        sorted_probs=pd.np.argsort(y_proba)
        probs=pd.np.sort(y_proba)
        print "highest 5, probs[-5:]", probs[-5:]

        reco_df=pd.DataFrame({'indices_desc':sorted_probs, 'probs':probs})
        reco_df.to_csv(DATAPATH+reco_file)


def write_topRecos(DATAPATH2, growth, testfile, reco_file, toprecofile):
    """
    writes toprecofile contains the top recommendations that are accepted
    :param DATAPATH2:
    :param growth: what percent of the recommendation are accepted
    :param testfile:  candidate pairs with hadamard distances
    :param reco_file: file containing indices pointing to top recommended pairs in testfile, along with the recommendation score
    :param toprecofile: output filename
    :return:
    """

    colsNames=['u1', 'u2', 'label', 'group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    for i in range(5):
        print "Iteration ", i
        DATAPATH=DATAPATH2+str(i)+"/"

        #nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')
        reco_df=pd.read_csv(DATAPATH+reco_file, index_col=0)
        test_df = pd.read_csv(DATAPATH + testfile, header=None, error_bad_lines=False, names=colsNames)

        reco_len=int(growth*(len(reco_df)))   #(nxgraph.number_of_edges()))
        print "reco_len, len(reco_df)", reco_len, len(reco_df)

        reco_indices=reco_df.iloc[-reco_len:, 0].values.tolist()
        probs=reco_df.iloc[-reco_len:, 1].values.tolist()
        reco_pairs=test_df.iloc[reco_indices,0:2].apply(tuple, axis=1).tolist()

        #G = nx.Graph()
        df=pd.DataFrame(((reco_pairs[i][0], reco_pairs[i][1],  probs[i]) for i in range(len(reco_pairs))), columns=['u', 'v', 'prob']  )
        df.to_csv(DATAPATH+ toprecofile, index=False)

        #reco_pairs[i][0], reco_pairs[i][1], weight= probs[i])
        #nx.write_edgelist(G, DATAPATH+ toprecofile)



def count_race_groups(race, DATAPATH2,sorted_recos, testfile, growtharr):
    #BY PRODUCT: makes race_testfile with race groups in the 4th column

    colsNames=['u1', 'u2', 'label', 'race_group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    for growth in growtharr:
        #print "growth ", growth

        group_count_dict = {k: [] for k in range(9)}
        arr_frac0, arr_frac1, arr_frac2=[], [], []

        for i in range(5):

            DATAPATH = DATAPATH2 + str(i) + "/"

            nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')
            test_df = pd.read_csv(DATAPATH + testfile, header=None, error_bad_lines=False, names=colsNames)


            test_df['race_group'] = test_df.apply(group_race, axis=1, args=(race,))
            test_df.to_csv(DATAPATH + 'race_'+testfile)

            sorted_recos_df=pd.read_csv(DATAPATH+sorted_recos, index_col=0)

            reco_len =int(growth * (nxgraph.number_of_edges()))
            reco_indices = sorted_recos_df.iloc[-reco_len:, 0].values.tolist()
            reco_test_df = test_df.iloc[reco_indices, 0:4]

            # group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8
            for gid in range(9):

                grp_cnt=len(reco_test_df[reco_test_df.race_group == gid])

                # if gid in [0, 3, 8] :
                #     group_count_dict[gid].append(grp_cnt*2)
                # else:
                group_count_dict[gid].append(grp_cnt)

            """temp=group_count_dict[1][-1]
            group_count_dict[1][-1] += group_count_dict[2][-1]
            group_count_dict[2][-1] += temp

            temp = group_count_dict[4][-1]
            group_count_dict[4][-1] += group_count_dict[6][-1]
            group_count_dict[6][-1] += temp


            temp = group_count_dict[5][-1]
            group_count_dict[5][-1] += group_count_dict[7][-1]
            group_count_dict[7][-1] += temp"""

            cnt_0 = len(reco_test_df[reco_test_df.race_group.isin([0, 2, 6])])
            cnt_1 = len(reco_test_df[reco_test_df.race_group.isin([1, 3, 7])])
            cnt_2 = len(reco_test_df[reco_test_df.race_group.isin([4, 5, 8])])
            summ =cnt_0+cnt_1+cnt_2

            #print cnt_0, cnt_1
            assert(summ ==len(reco_test_df))
            arr_frac0.append(float(cnt_0) / float(summ))
            arr_frac1.append(float(cnt_1) / float(summ))
            arr_frac2.append(float(cnt_2) / float(summ))

        print "growth, avg frac across 5 iterations for race 0,race1 race2"
        print growth, pd.np.average(arr_frac0), pd.np.average(arr_frac1), pd.np.average(arr_frac2)

        print "avg count across 5 iterations for each group: "
        print "growth, gid, avgcount"

        for gid in range(9):

            print growth, gid, pd.np.average(group_count_dict[gid])

        print "pd.np.average(group_count_dict[3])- pd.np.average(group_count_dict[0])", pd.np.average(group_count_dict[3])- pd.np.average(group_count_dict[0])
        print "pd.np.average(group_count_dict[3])- pd.np.average(group_count_dict[8])", pd.np.average(group_count_dict[3])- pd.np.average(group_count_dict[8])



def count_groups(DATAPATH2,sorted_recos, testfile, growtharr):

    colsNames=['u1', 'u2', 'label', 'group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    for growth in growtharr:
        #print "growth ", growth

        group_count_dict = {k: [] for k in range(4)}
        arr_frac0=[]

        for i in range(5):

            DATAPATH = DATAPATH2 + str(i) + "/"

            #nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')
            test_df = pd.read_csv(DATAPATH + testfile, header=None, error_bad_lines=False, names=colsNames)
            sorted_recos_df=pd.read_csv(DATAPATH+sorted_recos, index_col=0)

            reco_len =int(growth * len(sorted_recos_df))#(nxgraph.number_of_edges()))
            reco_indices = sorted_recos_df.iloc[-reco_len:, 0].values.tolist()
            #print "sorted_recos_df.shape, test_df.shape, len(reco_indices)"
            #print sorted_recos_df.shape, test_df.shape, len(reco_indices)
            reco_test_df = test_df.iloc[reco_indices, 0:4]


            # group: 00--0, 01--1, 10--2, 11--3
            for gid in [0, 1, 2, 3]:

                grp_cnt=len(reco_test_df[reco_test_df.group == gid])

                """if gid == 0 or gid == 3:
                    group_count_dict[gid].append(grp_cnt*2)
                else:"""
                group_count_dict[gid].append(grp_cnt)

            """temp=group_count_dict[1][-1]
            group_count_dict[1][-1] += group_count_dict[2][-1]
            group_count_dict[2][-1] += temp"""

            cnt_0= len(reco_test_df[reco_test_df.group.isin([0, 2])])
            cnt_1 = len(reco_test_df[reco_test_df.group.isin([1, 3])])

            assert(cnt_0+cnt_1==len(reco_test_df))
            arr_frac0.append(float(cnt_0)/float(cnt_0+cnt_1))

        print "growth, avg frac across 5 iterations for gender 0", growth, pd.np.average(arr_frac0)
        print "avg count and proportion across 5 iterations for each group: "
        print "growth, gid, avgcount, proportion"
        for gid in [0, 1, 2, 3]:
            print pd.np.average(group_count_dict[gid]), ","


def group_race_edglist(u1, u2, race):
    #group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8

    if (race[u1] == 0) & (race[u2] == 0): return  0
    if (race[u1] == 0) & (race[u2] == 1): return  1
    if (race[u1] == 1) & (race[u2] == 0): return  2
    if (race[u1] == 1) & ( race[u2]==1): return 3
    if (race[u1] == 0) & (race[u2] == 2): return  4
    if (race[u1] == 1) & (race[u2] == 2): return  5
    if (race[u1] == 2) & (race[u2] == 0): return  6
    if (race[u1] == 2) & ( race[u2]==1): return 7
    if (race[u1] == 2) & ( race[u2]==2): return 8

def group_gen_edglist(u1, u2, race):
    #group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8

    if (race[u1] == 0) & (race[u2] == 0): return  0
    if (race[u1] == 0) & (race[u2] == 1): return  1
    if (race[u1] == 1) & (race[u2] == 0): return  2
    if (race[u1] == 1) & ( race[u2]==1): return 3


def count_groups_org_edges_fullset(DATAPATH, gender, race):

        gen_count_dict = {k: [] for k in range(4)}
        race_count_dict = {k: [] for k in range(9)}

        nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')

        genders=[group_gen_edglist(u1, u2 , gender) for u1, u2 in nxgraph.edges()]

        for gid in range(4):
            gen_count_dict[gid].append(genders.count(gid))

        races=[group_race_edglist(u1, u2 , race) for u1, u2 in nxgraph.edges() ]

        for gid in range(9):
            race_count_dict[gid].append(races.count(gid))

        print "all possible across 5 iterations for each group: "

        for gid in [0,1,2,3]:
            print pd.np.average(gen_count_dict[gid]) , ", "

        for gid in range(9):
            print pd.np.average(race_count_dict[gid]) , ", "




def count_groups_org_edges(DATAPATH2, gender, race):

        gen_count_dict = {k: [] for k in range(4)}
        race_count_dict = {k: [] for k in range(9)}

        for i in range(5):

            DATAPATH = DATAPATH2 + str(i) + "/"
            nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')

            genders=[group_gen_edglist(u1, u2 , gender) for u1, u2 in nxgraph.edges() ]

            for gid in range(4):
                gen_count_dict[gid].append(genders.count(gid))

            races=[group_race_edglist(u1, u2 , race) for u1, u2 in nxgraph.edges() ]

            for gid in range(9):
                race_count_dict[gid].append(races.count(gid))

        print "all possible across 5 iterations for each group: "

        for gid in [0,1,2,3]:
            print pd.np.average(gen_count_dict[gid]) , ", "


        for gid in range(9):
            print pd.np.average(race_count_dict[gid]) , ", "




def count_groups_Organic(DATAPATH2, gender, race):

        group_count_dict = {k: [] for k in range(4)}
        race_count_dict = {k: [] for k in range(9)}

        for i in range(5):

            DATAPATH = DATAPATH2 + str(i) + "/"
            nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')

            genders=[gender[i] for i in nxgraph.nodes()]
            count_0=genders.count(0)
            count_1=genders.count(1)

            # group: 00--0, 01--1, 10--2, 11--3
            group_count_dict[0].append(count_0 * (count_0-1))
            group_count_dict[1].append(count_0 * count_1)
            group_count_dict[2].append(count_1 * count_0)
            group_count_dict[3].append(count_1 * (count_1-1))

            races=[race[n] for n in nxgraph.nodes()]

            count_0=races.count(0)
            count_1=races.count(1)
            count_2=races.count(2)

            # group: 00--0, 01--1, 10--2, 11--3, 02--4, 12--5, 20--6, 21--7, 22--8
            race_count_dict[0].append(count_0 * (count_0-1))
            race_count_dict[1].append(count_0 * count_1)
            race_count_dict[2].append(count_1 * count_0)
            race_count_dict[3].append(count_1 * (count_1-1))
            race_count_dict[4].append(count_0 * (count_2))
            race_count_dict[5].append(count_1 * count_2)
            race_count_dict[6].append(count_2 * count_0)
            race_count_dict[7].append(count_2 * (count_1))
            race_count_dict[8].append(count_2 * (count_2 - 1))


        print "all possible across 5 iterations for each group: "

        for gid in [0,1,2,3]:
            print pd.np.average(group_count_dict[gid]) , ", "


        for gid in range(9):
            print pd.np.average(race_count_dict[gid]) , ", "



def grow(DATAPATH2, growth, testfile, reco_file, grownfile):

    colsNames=['u1', 'u2', 'label', 'group']
    for i in range(0, 128):
        name = "comp_" + str(i)
        colsNames.append(name)

    for i in range(5):
        print "Iteration ", i
        DATAPATH=DATAPATH2+str(i)+"/"

        nxgraph = read_nxgraph(DATAPATH + 'known_80.edgelist')
        reco_df=pd.read_csv(DATAPATH+reco_file, index_col=0)
        reco_len=int(growth*(nxgraph.number_of_edges()))
        print "reco_len, nxgraph.number_of_edges()", reco_len, nxgraph.number_of_edges()
        reco_indices=reco_df.iloc[-reco_len:, 0].values.tolist()
        test_df  = pd.read_csv(DATAPATH +testfile, header=None, error_bad_lines=False, names=colsNames)
        print "test_df.columns[0:5]", test_df.columns[0:5]
        reco_pairs=test_df.iloc[reco_indices,0:2].apply(tuple, axis=1)
        nxgraph.add_edges_from(reco_pairs)
        nx.write_edgelist(nxgraph, DATAPATH + str(growth)+grownfile)
        print "after growing nxgraph.number_of_edges()", nxgraph.number_of_edges()




def glassCeil(DATAPATH2, binss, edgelist,  gc_File, organic, type2=False, ccdfflag=False):

    gender, race = read_instagram_genrace(DATAPATH2 + 'london.genrace')

    if not organic:

        for i in range(5):
            print "Iteration ", i
            DATAPATH=DATAPATH2+str(i)+"/"
            nxgraph = nx.read_edgelist(DATAPATH + edgelist, nodetype=int)#, delimiter=',',

            get_degree_dist(DATAPATH, nxgraph, binss, gender, gc_File, type2, ccdfflag)

    if organic:

        nxgraph =read_nxgraph(DATAPATH2 + edgelist)  # , delimiter=',',
        get_degree_dist(DATAPATH2, nxgraph, binss, gender, gc_File, type2, ccdfflag)




def glassCeil_race(DATAPATH2, binss, edgelist, gc_File, organic):
    genderdict, racedict = read_instagram_genrace(DATAPATH2 + 'london.genrace')

    if not organic:
        for it in range(5):
            print "Iteration ", it
            DATAPATH = DATAPATH2 + str(it) + "/"

            nxgraph = nx.read_edgelist(DATAPATH + edgelist, nodetype=int)  # , delimiter=',',
            get_degree_dist_race(DATAPATH2, nxgraph, binss, racedict, gc_File)


    elif organic:

        nxgraph = read_nxgraph(DATAPATH2 + edgelist)
        print DATAPATH2 + edgelist
        print "nxgraph.number_of_edges",  nxgraph.number_of_edges()

        get_degree_dist_race(DATAPATH2, nxgraph, binss, racedict, gc_File)




def glassCeil_race2(DATAPATH2, binss, edgelist, gc_File, organic, ccdf=False):
    genderdict, racedict = read_instagram_genrace(DATAPATH2 + 'london.genrace')

    if not organic:
        for it in range(5):
            print "Iteration ", it
            DATAPATH = DATAPATH2 + str(it) + "/"

            nxgraph = nx.read_edgelist(DATAPATH + edgelist, nodetype=int)  # , delimiter=',',

            get_degree_dist_race2(DATAPATH, nxgraph, binss, racedict, gc_File, ccdf)

    elif organic:

        nxgraph = read_nxgraph(DATAPATH2 + edgelist)
        print DATAPATH2 + edgelist
        print "nxgraph.number_of_edges",  nxgraph.number_of_edges()

        get_degree_dist_race2(DATAPATH2, nxgraph, binss, racedict, gc_File, ccdf)





def get_degree_dist(DATAPATH,nxgraph, binss, gender, gc_File, type2, ccdfflag):

    degree_sequence_1 = sorted([d for n, d in nxgraph.degree() if gender[n] == 1], reverse=True)  # degree sequence
    degree_sequence = sorted([d for n, d in nxgraph.degree() if gender[n] == 0], reverse=True)  # degree sequence

    print "Degree sequence_1", degree_sequence_1[:10]
    print "Degree sequence", degree_sequence[:10]

    gen0, gen1 = len(degree_sequence), len(degree_sequence_1)

    deg_bin_0 = pd.cut(degree_sequence, bins=binss, include_lowest=True)
    # ax = deg_bin_0.value_counts().plot.bar(rot=0, color="b")
    bin_0 = deg_bin_0.value_counts().values

    deg_bin_1 = pd.cut(degree_sequence_1, bins=binss, include_lowest=True)
    # ax = deg_bin_1.value_counts().plot.bar(rot=0, color="r")
    bin_1 = deg_bin_1.value_counts().values

    cats = deg_bin_1.value_counts().index.categories
    print "range, count_gen_0,  count_gen_1,  frac_gen_0,  frac_gen_1, count_total"
    print "(", min(min(degree_sequence), min(degree_sequence_1)), max(max(degree_sequence), max(
        degree_sequence_1)), "]", gen0, gen1, float(gen0) / float(gen0 + gen1), float(gen1) / float(
        gen1 + gen0), gen1 + gen0

    arr = []
    for i in range(len(binss) - 1):
        try:
            if ccdfflag:
                arr.append([cats[i], bin_0[i]])

            else:
                if not type2:
                    frac_0 = float(bin_0[i]) / float((bin_0[i] + bin_1[i]))
                else:
                    frac_0 = float(bin_0[i]) / float(gen0)

                arr.append([cats[i], frac_0])
            print  cats[i], bin_0[i], bin_1[i], frac_0, float(bin_1[i]) / float((bin_0[i] + bin_1[i])), ( bin_0[i] + bin_1[i])
        except:
            continue

    df = pd.DataFrame(data=arr, columns=['range', 'frac_0'])
    df.to_csv(DATAPATH + gc_File)



def get_degree_dist_race2(DATAPATH, nxgraph, binss, racedict, gc_File, ccdf):

    degree_sequence_1 = sorted([d for n, d in nxgraph.degree() if racedict[n] == 1], reverse=True)  # degree sequence
    degree_sequence = sorted([d for n, d in nxgraph.degree() if racedict[n] == 0], reverse=True)  # degree sequence
    degree_sequence_2 = sorted([d for n, d in nxgraph.degree() if racedict[n] == 2], reverse=True)

    print "Degree sequence_1[:10]", degree_sequence_1[:10]
    print "Degree sequence[:10]", degree_sequence[:10]
    print "Degree sequence_2[:10]", degree_sequence_2[:10]

    gen0, gen1, gen2 = len(degree_sequence), len(degree_sequence_1), len(degree_sequence_2)

    deg_bin_0 = pd.cut(degree_sequence, bins=binss, include_lowest=True)
    # ax = deg_bin_0.value_counts().plot.bar(rot=0, color="b")
    bin_0 = deg_bin_0.value_counts().values

    deg_bin_1 = pd.cut(degree_sequence_1, bins=binss, include_lowest=True)
    # ax = deg_bin_1.value_counts().plot.bar(rot=0, color="r")
    bin_1 = deg_bin_1.value_counts().values

    deg_bin_2 = pd.cut(degree_sequence_2, bins=binss, include_lowest=True)
    # ax = deg_bin_1.value_counts().plot.bar(rot=0, color="r")
    bin_2 = deg_bin_2.value_counts().values

    cats = deg_bin_1.value_counts().index.categories
    print "range, count_gen_0,  count_gen_1, count_gen_2, frac_gen_0,  frac_gen_1,  frac_gen_2, count_total"
    print "(", min(min(degree_sequence), min(degree_sequence_1), min(degree_sequence_2)), max(max(degree_sequence),
                                                                                              max(degree_sequence_1),
                                                                                              max(
                                                                                                  degree_sequence_2)), "]", gen0, gen1, gen2, float(
        gen0) / float(gen0 + gen1 + gen2), float(gen1) / float(
        gen1 + gen0), float(gen2) / float(gen0 + gen1 + gen2), gen1 + gen0 + gen2

    arr = []
    for i in range(len(binss) - 1):
        try:
            sum_i = (bin_0[i] + bin_1[i] + bin_2[i])
            frac_0 = float(bin_0[i]) / gen0
            frac_1 = float(bin_1[i]) / gen1
            frac_2 = float(bin_2[i]) / gen2
            print cats[i], bin_0[i], bin_1[i], bin_2[i], frac_0, frac_1, frac_2, sum_i

            if ccdf:
                arr.append([cats[i], bin_0[i], bin_1[i],bin_2[i],])
            elif not ccdf:
                arr.append([cats[i], frac_0, frac_1, frac_2])

        except:
            continue

    df = pd.DataFrame(data=arr, columns=['range', 'frac_0', 'frac_1', 'frac_2'])
    df.to_csv(DATAPATH + gc_File)


def get_degree_dist_race(DATAPATH, nxgraph, binss, racedict, gc_File):

    degree_sequence_1 = sorted([d for n, d in nxgraph.degree() if racedict[n] == 1], reverse=True)  # degree sequence
    degree_sequence = sorted([d for n, d in nxgraph.degree() if racedict[n] == 0], reverse=True)  # degree sequence
    degree_sequence_2 = sorted([d for n, d in nxgraph.degree() if racedict[n] == 2], reverse=True)

    print "Degree sequence_1[:10]", degree_sequence_1[:10]
    print "Degree sequence[:10]", degree_sequence[:10]
    print "Degree sequence_2[:10]", degree_sequence_2[:10]

    gen0, gen1, gen2 = len(degree_sequence), len(degree_sequence_1), len(degree_sequence_2)

    deg_bin_0 = pd.cut(degree_sequence, bins=binss, include_lowest=True)
    # ax = deg_bin_0.value_counts().plot.bar(rot=0, color="b")
    bin_0 = deg_bin_0.value_counts().values

    deg_bin_1 = pd.cut(degree_sequence_1, bins=binss, include_lowest=True)
    # ax = deg_bin_1.value_counts().plot.bar(rot=0, color="r")
    bin_1 = deg_bin_1.value_counts().values

    deg_bin_2 = pd.cut(degree_sequence_2, bins=binss, include_lowest=True)
    # ax = deg_bin_1.value_counts().plot.bar(rot=0, color="r")
    bin_2 = deg_bin_2.value_counts().values

    cats = deg_bin_1.value_counts().index.categories
    print "range, count_gen_0,  count_gen_1, count_gen_2, frac_gen_0,  frac_gen_1,  frac_gen_2, count_total"
    print "(", min(min(degree_sequence), min(degree_sequence_1), min(degree_sequence_2)), max(max(degree_sequence),
                                                                                              max(degree_sequence_1),
                                                                                              max(
                                                                                                  degree_sequence_2)), "]", gen0, gen1, gen2, float(
        gen0) / float(gen0 + gen1 + gen2), float(gen1) / float(
        gen1 + gen0), float(gen2) / float(gen0 + gen1 + gen2), gen1 + gen0 + gen2

    arr = []
    for i in range(len(binss) - 1):
        try:
            sum_i = (bin_0[i] + bin_1[i] + bin_2[i])
            frac_0 = float(bin_0[i]) / float(sum_i)
            frac_1 = float(bin_1[i]) / float(sum_i)
            frac_2 = float(bin_2[i]) / float(sum_i)
            print  cats[i], bin_0[i], bin_1[i], bin_2[i], frac_0, frac_1, frac_2, sum_i
            arr.append([cats[i], frac_0, frac_1, frac_2])
        except:
            continue

    df = pd.DataFrame(data=arr, columns=['range', 'frac_0', 'frac_1', 'frac_2'])
    df.to_csv(DATAPATH + gc_File)


def get_biasDist_genrace(gender, race, DATAPATH2, recofile, biasfile):

    for it in range(5):
        print "Iteration ", it
        DATAPATH = DATAPATH2 + str(it) + "/"

        df = pd.read_csv(DATAPATH + recofile)


        grouped = df.groupby(['u'])

        arr = []

        for u, group in grouped:
            in_arr = [u, gender[u], race[u]]

            num_v, race_K = 0, 0
            fem_ctr, race_ctr_0, race_ctr_1, race_ctr_2 = 0.0, 0.0, 0.0, 0.0

            for u2 in group.v:

                if gender[u2] != -1:
                    num_v += 1

                    if (gender[u2] == 1):  # if female friend
                        fem_ctr += 1

                if race[u2] != -1:
                    race_K += 1

                    if (race[u2] == 0):
                        race_ctr_0 += 1

                    if (race[u2] == 1):
                        race_ctr_1 += 1

                    if (race[u2] == 2):
                        race_ctr_2 += 1

            if num_v == 0 or gender[u] == -1:
                fem_ctr = -1

            else:
                fem_ctr /= num_v


            if race_K == 0 or race[u] == -1:
                race_ctr_0, race_ctr_1, race_ctr_2 =  -1, -1, -1
            else:
                race_ctr_0 /= race_K
                race_ctr_1 /= race_K
                race_ctr_2 /= race_K

            in_arr.extend([fem_ctr, race_ctr_0, race_ctr_1, race_ctr_2])

            arr.append(in_arr)

        bias_df = pd.DataFrame(data=arr, columns=['u', 'gender[u]', 'race[u]',
                                                  'sup_fem_gen_bias',
                                                  'sup_0_race_bias', 'sup_1_race_bias', 'sup_2_race_bias',
                                                 ])

        bias_df.to_csv(DATAPATH + biasfile)



######## CURRENTLY UNUSED FUNCTIONS ########


def get_accuracies(DATAPATH, testfile_suffix, trainfile_suffix):
    acc_arr=[]

    for i in range(5):
        print "Iteration ", i
        row = pd.read_csv(DATAPATH + i + testfile_suffix, header=None, error_bad_lines=False)  # , index_col=0)
        print test_df.shape, "test_df.shape"
        test_df.dropna(inplace=True)
        print test_df.shape, "test_df.shape after dropna"
        df = pd.DataFrame()
        df['u'] = test_df.iloc[:, 0].values
        df['v'] = test_df.iloc[:, 1].values
        y_test= test_df.iloc[:, 2].values
        X_test=test_df.iloc[:, 3:].values

        clf=trainRF(DATAPATH + i + trainfile_suffix)
        print "RF trained"

        acc = clf.score(X_test,y_test)
        print "acc", acc
        acc_arr.append(acc)


    acc_df=pd.DataFrame({'acc',acc_arr})
    acc_df.to_csv(DATAPATH+"accuracies_unfair.csv")







def get_bias_genrace(DATAPATH, recofile, k_arr):


    df = pd.read_csv(DATAPATH+ recofile)

    gender, race = read_instagram_genrace(DATAPATH+"london.genrace")

    grouped = df.groupby(['u'])

    arr = []
    count_df_arr=[]

    for u, group in grouped:
        in_arr=[u, gender[u], race[u]]

        for k in k_arr:

            if gender[u] == -1 and race[u] == -1:
                in_arr.extend([-1, -1, -1, -1, -1, -1])
                arr.append(in_arr)
                continue


            gen_K, race_K = 0, 0
            same_gen_ctr, fem_ctr, same_race_ctr , race_ctr_0,  race_ctr_1,  race_ctr_2 = 0.0, 0.0,  0.0, 0.0,  0.0, 0.0

            for u2 in group.v:

                if gender[u2]!=-1:
                    gen_K+=1

                    if (gen_K==k):
                        break

                    if (gender[u] == gender[u2]):
                        same_gen_ctr += 1

                    if (gender[u2]==1): #if female friend
                        fem_ctr+=1

                if race[u2] !=-1:
                    race_K+=1

                    if (race_K==k):
                        break

                    if (race[u] == race[u2]):
                        same_race_ctr += 1

                    if (race[u2] == 0):
                        race_ctr_0 += 1

                    if (race[u2] == 1):
                        race_ctr_1 += 1

                    if (race[u2] == 2):
                        race_ctr_2 += 1




            if gen_K==0 or gender[u]==-1:
                same_gen_ctr,fem_ctr =-1, -1
            else:
                same_gen_ctr /= gen_K
                fem_ctr /= gen_K

            if race_K==0 or race[u]==-1:
                same_race_ctr, race_ctr_0, race_ctr_1, race_ctr_2= -1, -1, -1, -1
            else:
                same_race_ctr /=race_K
                race_ctr_0 /= race_K
                race_ctr_1 /=race_K
                race_ctr_2 /= race_K


            in_arr.extend([same_gen_ctr, fem_ctr, same_race_ctr, race_ctr_0 , race_ctr_1,  race_ctr_2])
            count_df_arr.append([u, k, gen_K, race_K])

        arr.append(in_arr)

    bias_df = pd.DataFrame(data=arr, columns=['u', 'gender[u]', 'race[u]',
         'sup_same_gen_bias100','sup_fem_gen_bias100',   'sup_same_race_bias100', 'sup_0_race_bias100','sup_1_race_bias100', 'sup_2_race_bias100',
         'sup_same_gen_bias50','sup_fem_gen_bias50',     'sup_same_race_bias50','sup_0_race_bias50','sup_1_race_bias50', 'sup_2_race_bias50',
         'sup_same_gen_bias10','sup_fem_gen_bias10' ,    'sup_same_race_bias10','sup_0_race_bias10','sup_1_race_bias10', 'sup_2_race_bias10'])


    bias_df.to_csv(DATAPATH+"sup_bias_250.csv")


    count_df= pd.DataFrame(data=count_df_arr, columns=['u', 'top_k', 'gen_K', 'race_K'])
    count_df.to_csv(DATAPATH+ "size_non-neg_users_250.csv")



def get_bias_json(DATAPATH, k_arr):

    for filename in ["top100cos.json", "top100euc.json"]:
        arr = read_json(DATAPATH+filename)

        gender, race = read_instagram_genrace(DATAPATH + "london.genrace")

        outer_arr = []
        count_df_arr=[]
        for u in range(len(arr)):
            in_arr = [u, gender[u], race[u]]
            for k in k_arr:
                gen_K, race_K = 0, 0
                same_gen_ctr, fem_ctr, same_race_ctr, race_ctr_0, race_ctr_1, race_ctr_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for u2, prob in arr[u]:

                    if gender[u2] != -1:
                        gen_K += 1

                        if (gen_K == k):
                            break

                        if (gender[u] == gender[u2]):
                            same_gen_ctr += 1

                        if (gender[u2] == 1):  # if female friend
                            fem_ctr += 1

                    if race[u2] != -1:
                        race_K += 1

                        if (race_K == k):
                            break

                        if (race[u] == race[u2]):
                            same_race_ctr += 1

                        if (race[u2] == 0):
                            race_ctr_0 += 1

                        if (race[u2] == 1):
                            race_ctr_1 += 1

                        if (race[u2] == 2):
                            race_ctr_2 += 1

                same_gen_ctr /= gen_K
                fem_ctr /= gen_K
                same_race_ctr /= race_K
                race_ctr_0 /= race_K
                race_ctr_1 /= race_K
                race_ctr_2 /= race_K
                in_arr.extend([same_gen_ctr, fem_ctr, same_race_ctr, race_ctr_0, race_ctr_1, race_ctr_2])
                count_df_arr.append([u, k, gen_K, race_K])

            outer_arr.append(in_arr)

        bias_df = pd.DataFrame(data=arr, columns=['u', 'gender[u]', 'race[u]',
                                                  'sup_same_gen_bias100', 'sup_fem_gen_bias100',
                                                  'sup_same_race_bias100', 'sup_0_race_bias100', 'sup_1_race_bias100',
                                                  'sup_2_race_bias100',
                                                  'sup_same_gen_bias50', 'sup_fem_gen_bias50', 'sup_same_race_bias50',
                                                  'sup_0_race_bias50', 'sup_1_race_bias50', 'sup_2_race_bias50',
                                                  'sup_same_gen_bias10', 'sup_fem_gen_bias10', 'sup_same_race_bias10',
                                                  'sup_0_race_bias10', 'sup_1_race_bias10', 'sup_2_race_bias10'])

        bias_df.to_csv(DATAPATH + "unsup_bias.csv")

        count_df = pd.DataFrame(data=count_df_arr, columns=['u', 'top_k', 'gen_K', 'race_K'])
        count_df.to_csv(DATAPATH + "size_non-neg_users.csv")




