import sys
sys.path.append("../../src/")
from graph_utils import read_instagram_genrace
from dataset_utils import make_5_testfiles#, make_5_recofiles#, make_5_trainfiles, make_5_testfiles,
from bias_metrics import get_biasDist_genrace, write_topRecos, count_groups, sort_Reco, getROCs, grow#, get_bias_genrace,get_degree_dist, get_accuracies, get_Equality_opportunity, get_Disparate_impact,
#from plot_utils import plot_bias

DATAPATH= "../../data/la/"

gender, race = read_instagram_genrace(DATAPATH + 'la.genrace')
"""

#make_5_trainfiles(gender, DATAPATH)

##### Branch 1 ##########

make_5_recofiles(gender, DATAPATH,embfile='known_80_80_20_128.emb', pickfile_frn="known_80_top100.pick", pickfile_str="known_80_rand.pick", testfile_suffix= "hada_100_test.csv")
make_5_recofiles(gender, DATAPATH,embfile='known_80_gendeq_80_20_128.emb', pickfile_frn="known_80_gendeq_top100.pick" , pickfile_str="known_80_rand.pick", testfile_suffix= "hada_100_test_geneq.csv")


sort_Reco(DATAPATH, testfile= "hada_100_test.csv", reco_file= "recommendations.csv")
sort_Reco(DATAPATH, testfile= "hada_100_test_geneq.csv", reco_file= "recommendations_geneq.csv")


write_topRecos(DATAPATH, growtharr=[0.2], testfile="hada_100_test.csv", reco_file= "recommendations.csv", toprecofile="topRecos.edgelist")
write_topRecos(DATAPATH, growtharr=[0.2], testfile="hada_100_test_geneq.csv", reco_file= "recommendations_geneq.csv", toprecofile="topRecos_geneq.edgelist")

print "#######################"
print "REGULAR"
print "#######################"
count_groups(DATAPATH,sorted_recos="recommendations.csv", testfile="hada_100_test.csv", growtharr=[0.20, 0.10, 0.05])

print "#######################"
print "FAIRWALK"
print "#######################"
count_groups(DATAPATH,sorted_recos="recommendations_geneq.csv", testfile="hada_100_test_geneq.csv", growtharr=[  0.10, 0.05])



get_biasDist_genrace(gender, race, DATAPATH, recofile="topRecos.edgelist", biasfile="biases.csv")

get_biasDist_genrace(gender, race, DATAPATH, recofile="topRecos_geneq.edgelist", biasfile="biases_geneq.csv")

plot_bias(DATAPATH, "distbias.png")



##### Branch 1.1 ##########

grow(DATAPATH, growth=0.25, testfile="hada_100_test.csv", reco_file= "recommendations.csv", grownfile="_grown.edgelist")
grow(DATAPATH, growth=0.25, testfile="hada_100_test_geneq.csv", reco_file= "recommendations_geneq.csv", grownfile="_grown_gen_eq.edgelist")


##### Branch 1.2 ##########

# Barteks distribution stuff

#get_recommmendation_statistics()
#plot_recommendation_statistics()
"""

##### Branch 2 ##########

make_5_testfiles(gender, DATAPATH, embfile='known_80_80_20_128.emb', edgelist='known_20.edgelist', pickfile_str="known_80_rand.pick", testfile_suffix= "hada_test_new.csv")
make_5_testfiles(gender, DATAPATH, embfile='known_80_gendeq_80_20_128.emb', edgelist='known_20.edgelist', pickfile_str="known_80_rand.pick", testfile_suffix= "hada_test_geneq_new.csv")


ROC_df=getROCs(DATAPATH, rocfile_suffix="_ROC_df_new.csv", testfile_suffix='hada_test_new.csv')
ROC_df=getROCs(DATAPATH, rocfile_suffix="_ROC_df_geneq_new.csv", testfile_suffix='hada_test_geneq_new.csv')
