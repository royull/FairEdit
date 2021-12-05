import sys
sys.path.append("../../src/")
from graph_utils import read_instagram_genrace
from dataset_utils import make_5_recofiles #, make_5_trainfilesmake_5_testfiles,
from bias_metrics import sort_Reco, write_topRecos,  count_race_groups  # ,  getROCs_race, grow, glassCeil,  get_degree_dist  # , get_bias_genrace,get_degree_dist, get_accuracies, get_Equality_opportunity, get_Disparate_impact


DATAPATH= "../../data/london/"


gender, race = read_instagram_genrace(DATAPATH + 'london.genrace')


#make_5_trainfiles(gender, DATAPATH)

##### Branch 1 ##########

#make_5_recofiles(gender, DATAPATH,embfile='known_80_80_20_128.emb', pickfile_frn="known_80_top100.pick", pickfile_str="known_80_rand.pick", testfile_suffix= "hada_100_test_2.csv")
make_5_recofiles(gender, DATAPATH,embfile='known_80_raceeq_80_20_128.emb', pickfile_frn="known_80_raceeq_top100.pick", pickfile_str="known_80_rand.pick", testfile_suffix= "hada_100_test_raceeq_2.csv")

#sort_Reco(DATAPATH, testfile= "hada_100_test.csv", reco_file= "recommendations_1.csv")
sort_Reco(DATAPATH, testfile= "hada_100_test_raceeq_2.csv", reco_file= "recommendations_raceeq.csv")


#write_topRecos(DATAPATH, growtharr=[0.20], testfile="hada_100_test_2.csv", reco_file= "recommendations.csv", toprecofile="_topRecos.edgelist")
write_topRecos(DATAPATH, growtharr=[0.20], testfile="hada_100_test_raceeq_2.csv", reco_file= "recommendations_raceeq.csv", toprecofile="topRecos_raceeq.edgelist")


print "#######################"
print "REGULAR"
print "#######################"
count_race_groups(race, DATAPATH, sorted_recos="recommendations.csv", testfile="hada_100_test_2.csv", growtharr=[0.2])# 0.1, 0.05])
#makes race_testfile with race groups in the 4th column

print "#######################"
print "FAIRWALK"
print "#######################"
count_race_groups(race, DATAPATH, sorted_recos="recommendations_raceeq.csv", testfile="hada_100_test_raceeq_2.csv", growtharr=[0.2])#1, 0.05])



##### Branch 1.1 ##########

#grow(DATAPATH, growth=0.25, testfile="hada_100_test.csv", reco_file= "recommendations.csv", grownfile="_grown.edgelist")
#grow(DATAPATH, growth=0.25, testfile="hada_100_test_raceeq.csv", reco_file= "recommendations_raceeq.csv", grownfile="_grown_raceeq.edgelist")


##### Branch 1.2 ##########

# distribution stuff

##### Branch 2 ##########

make_5_testfiles(gender, DATAPATH, embfile='known_80_80_20_128.emb', edgelist='known_20.edgelist', pickfile_str="known_80_rand.pick", testfile_suffix= "hada_test_new.csv")
make_5_testfiles(gender, DATAPATH, embfile='known_80_raceeq_80_20_128.emb', edgelist='known_20.edgelist', pickfile_str="known_80_rand.pick", testfile_suffix= "hada_test_raceeq.csv")

ROC_df=getROCs_race(race, DATAPATH, rocfile_suffix="_ROC_df_race.csv", testfile_suffix='hada_test_new.csv')
ROC_df=getROCs_race(race, DATAPATH, rocfile_suffix="_ROC_df_raceeq.csv", testfile_suffix='hada_test_raceeq.csv')
