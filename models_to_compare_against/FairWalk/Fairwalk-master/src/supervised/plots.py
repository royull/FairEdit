
# Plots some of the plots in the paper

from bias_metrics import  glassCeil_race2,  glassCeil_race,  glassCeil
from plot_utils import  plotGC_race_together, plotGC, plotROCs, plotGC_race
DATAPATH='../../data/london/'

##### RACE ######

step=2
binss = xrange(0, 20, step)   # [1,10, 20, 50, 75, 100, 125, 150, 175, 200, 500]

orgfile=str(step)+"_race_GC_degreeDist_org.csv"
fairfile= str(step)+"_race_GC_degreeDist_raceeq.csv"
regfile=str(step)+"_race_GC_degreeDist_reg.csv"


#glassCeil_race2(DATAPATH,  binss,  edgelist= "0.25_grown.edgelist",   gc_File=regfile , organic=False)
#glassCeil_race2(DATAPATH,  binss, edgelist= "0.25_grown_raceeq.edgelist", gc_File=fairfile, organic=False)
glassCeil_race2(DATAPATH,  binss,  edgelist= "london_known.edgelist",  gc_File=orgfile , organic=True)

plotGC_race(DATAPATH, orgfile, regfile, fairfile, plotfile="CCDF_GC_"+ str(step)+ "step_race_separate.png")
plotGC_race_together(DATAPATH, orgfile, regfile, fairfile, plotfile="CCDF_GC_"+ str(step)+ "step_race_faironly.png")


orgfile=str(step)+"_race_GC_degreeDist_org_Ccdf.csv"
fairfile= str(step)+"_race_GC_degreeDist_raceeq_Ccdf.csv"
regfile=str(step)+"_race_GC_degreeDist_reg_Ccdf.csv"

#glassCeil_race2(DATAPATH,  binss,  edgelist= "0.25_grown.edgelist",   gc_File=regfile , organic=False, ccdf=True)
#glassCeil_race2(DATAPATH,  binss, edgelist= "0.25_grown_raceeq.edgelist", gc_File=fairfile, organic=False, ccdf=True)
glassCeil_race2(DATAPATH,  binss,  edgelist= "london_known.edgelist",  gc_File=orgfile , organic=True, ccdf=True)

#plotGC_race(DATAPATH, orgfile, regfile, fairfile, plotfile="CCDF_GC_"+ str(step)+ "step_race_separate_ccdf.png")
plotGC_race_together(DATAPATH, orgfile, regfile, fairfile, plotfile="CCDF_GC_"+ str(step)+ "step_race_faironly_ccdf.png", ccdfflag=True)




glassCeil_race(DATAPATH,  binss,  edgelist= "0.25_grown.edgelist",   gc_File=regfile , organic=False)
glassCeil_race(DATAPATH,  binss, edgelist= "0.25_grown_raceeq.edgelist", gc_File=fairfile, organic=False)
glassCeil_race(DATAPATH,  binss,  edgelist= "london_known.edgelist",  gc_File=orgfile , organic=True)

plotGC_race(DATAPATH, orgfile, regfile, fairfile, plotfile="GC_"+ str(step)+ "step_race.png")

plotROCs(DATAPATH + "ROCS_race/",  window_size = 21, polynomial_order =0, plotfile_pref='../plots/ROC_race_', rocfile_suff="_ROC_df_race.csv" , race=True)
plotROCs(DATAPATH + "ROCS_race/",  window_size = 21, polynomial_order =0, plotfile_pref='../plots/ROC_raceeq_', rocfile_suff="_ROC_df_raceeq.csv", race=True)

##### GENDER ######

step=2
binss =xrange(0,50, step)   # [1,10, 20, 50, 75, 100, 125, 150, 175, 200, 500]


orgfile=str(step)+"_gen_GC_degreeDist_org_Ccdf.csv"
fairfile= str(step)+"_gen_GC_degreeDist_geneq_Ccdf.csv"
regfile=str(step)+"_gen_GC_degreeDist_reg_Ccdf.csv"

glassCeil(DATAPATH,  binss,  edgelist= "0.1_grown.edgelist",   gc_File=regfile , organic=False , type2=True, ccdfflag=True)
glassCeil(DATAPATH,  binss, edgelist= "0.1_grown_gen_eq.edgelist", gc_File=fairfile, organic=False, type2=True,ccdfflag=True)
glassCeil(DATAPATH,  binss,  edgelist= "london_known.edgelist",    gc_File=orgfile , organic=True, type2=True,ccdfflag=True)


plotGC(DATAPATH, orgfile, regfile, fairfile, plotfile= "CCDF_GC_"+ str(step)+ "step_gender.png", ccdfflag=True)
"""
orgfile=str(step)+"_gen_GC_degreeDist_org.csv"
fairfile= str(step)+"_gen_GC_degreeDist_geneq.csv"
regfile=str(step)+"_gen_GC_degreeDist_reg.csv"

glassCeil(DATAPATH,  binss,  edgelist= "0.25_grown.edgelist",   gc_File=regfile , organic=False)
glassCeil(DATAPATH,  binss, edgelist= "0.25_grown_gen_eq.edgelist", gc_File=fairfile, organic=False)
glassCeil(DATAPATH,  binss,  edgelist= "london_known.edgelist",    gc_File=orgfile , organic=True)

plotGC(DATAPATH, orgfile, regfile, fairfile, plotfile= "GC_"+ str(step)+ "step_gender.png", ccdfflag=False)

#plotROCs(DATAPATH + "ROCS/",  window_size = 101, polynomial_order =0, plotfile_pref='../plots/ROC_gen_', rocfile_suff="_ROC_df_new.csv", race=False)
#plotROCs(DATAPATH + "ROCS/",  window_size = 101, polynomial_order =0, plotfile_pref='../plots/ROC_geneq_', rocfile_suff="_ROC_df_geneq_new.csv", race =False)
"""