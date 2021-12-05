import sys
sys.path.append("../../src/")
from graph_utils import read_instagram_genrace
from bias_metrics import getROCs_race,  getROCs # get_degree_dist  # , get_bias_genrace,get_degree_dist, get_accuracies, get_Equality_opportunity, get_Disparate_impact


DATAPATH= "../../data/london/"

gender, race = read_instagram_genrace(DATAPATH + 'london.genrace')
#make_5_trainfiles(gender, DATAPATH)

ROC_df=getROCs( DATAPATH, rocfile_suffix="_ROC_df_new.csv", testfile_suffix='hada_test_new.csv')
ROC_df=getROCs(DATAPATH, rocfile_suffix="_ROC_df_geneq_new.csv", testfile_suffix='hada_test_raceeq.csv')



ROC_df=getROCs_race(race, DATAPATH, rocfile_suffix="_ROC_df_race.csv", testfile_suffix='hada_test_new.csv')
ROC_df=getROCs_race(race, DATAPATH, rocfile_suffix="_ROC_df_raceeq.csv", testfile_suffix='hada_test_raceeq.csv')

