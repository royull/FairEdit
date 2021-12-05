import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import pandas as pd
import re
from numpy.ma import arange
import numpy as np
from scipy.signal import savgol_filter
import json

def set_font():
    plt.rcParams.update({'font.size': 31, 'lines.linewidth': 4})
    plt.rcParams['text.usetex'] = True

set_font()
font_title_size = 34
color_network = '#b2abd2'
color_standard = '#e66101'
color_fair = '#5e3c99'
alpha = 0.7
fig_width = 15
ang=45

def plotQualitysamecol(prec_reg, prec_geneq, prec_race, recall_reg, recall_geneq, recall_race):

    fig, ax1 = plt.subplots(figsize=(fig_width,fig_width/15.*9))
    t=[1e4, 5e4, 10e4, 25e4, 50e4, 75e4, 100e4]#,1058589]
    #color = 'tab:blue'
    ax1.set_xlabel('number of recommendations')
    ax1.set_ylabel('recall')#, color=color)
    ax1.plot(t, recall_reg[:-1],  'x-', color='#e66101',  label='recall regular', linestyle='--')
    ax1.plot(t, recall_geneq[:-1], 'xb-', label='recall fair gender', linestyle='--')
    ax1.plot(t, recall_race[:-1], 'xg-',  label='recall fair race', linestyle='--',)
    ax1.tick_params(axis='y')#, labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xticklabels([r'$1.0\mathrm{e}^4$', r'$5.0\mathrm{e}^4$', r'$1.0\mathrm{e}^5$', r'$2.5\mathrm{e}^5$',r'$5.0\mathrm{e}^5$', r'$7.5\mathrm{e}^5$', r'$1.0\mathrm{e}^6$'])#,rotation=ang)

    #color = 'tab:green'
    ax2.set_ylabel('precision')#, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, prec_reg[:-1],  '^-',color='#e66101',  label='precision regular')
    ax2.plot(t, prec_geneq[:-1], '^b-', label='precision fair gender')
    ax2.plot(t, prec_race[:-1], '^g-',  label='precision fair race')
    ax2.tick_params(axis='y')#, labelcolor=color)
    ax2.set_ylim(0, 0.04)

    plt.figlegend(loc='upper center', ncol=2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.subplots_adjust( top=0.73, bottom=0.11)#hspace=0.32, wspace=0.05, right=0.9,

    plt.savefig("quality.pdf", format='pdf')
    plt.show()


def plotQuality(prec_reg, prec_geneq, prec_race, recall_reg, recall_geneq, recall_race):

    fig, ax1 = plt.subplots()
    t=[10000, 50000, 100000, 250000, 500000, 750000, 1000000]#,1058589]
    color = 'tab:blue'
    ax1.set_xlabel('cut-off')
    ax1.set_ylabel('precision', color=color)
    ax1.plot(t, prec_reg,  '.r-',  label='precision regular')
    ax1.plot(t,prec_geneq, 'xg-',  label='precision fair_gender')
    ax1.plot(t, prec_race, '^b-', label='precision fair_race')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, recall_reg,  'sr-',  label='recall regular')
    ax2.plot(t,recall_geneq, 'xb-',  label='recall fair_gender')
    ax2.plot(t, recall_race, '^g-',  label='recall fair_race')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.figlegend(loc='upper center', ncol=2)

    fig.tight_layout(pad=2)  # otherwise the right y-label is slightly clipped
    plt.show()


def plotcombi_ER(la_fair_f, lon_fair_f, la_reg_f,  lon_reg_f,la_fair_f_race, lon_fair_f_race, la_reg_f_race,  lon_reg_f_race):

    X = pd.np.arange(4)
    fig, axarr = plt.subplots(2, 2,figsize=(fig_width,fig_width/15.*12))
    axarr[0, 0].bar(X, la_reg_f, color='#e66101', alpha=alpha, width=0.40, label='regular')
    axarr[0, 0].bar(X + 0.40, la_fair_f, color='#5e3c99',alpha=alpha, width=0.40, label='fair')
    axarr[0, 0].set_title('LA - gender', {'fontsize': font_title_size})
    axarr[0, 1].bar(X, lon_reg_f, color='#e66101', alpha=alpha, width=0.40)
    axarr[0, 1].bar(X + 0.40, lon_fair_f, color='#5e3c99',alpha=alpha, width=0.40)
    axarr[0, 1].set_title('London - gender', {'fontsize': font_title_size})

    ### RACE ###

    Xr = pd.np.arange(9)
    axarr[1, 0].bar(Xr, la_reg_f_race, color='#e66101',alpha=alpha, width=0.30)
    axarr[1, 0].bar(Xr + 0.30, la_fair_f_race, color='#5e3c99',alpha=alpha, width=0.30)
    axarr[1, 0].set_title('LA - race', {'fontsize': font_title_size})
    axarr[1, 1].bar(Xr, lon_reg_f_race, color='#e66101',alpha=alpha, width=0.30)
    axarr[1, 1].bar(Xr + 0.30, lon_fair_f_race, color='#5e3c99', alpha=alpha, width=0.30)
    axarr[1, 1].set_title('London - race', {'fontsize': font_title_size})

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.setp(axarr[0,:], xticks=pd.np.asarray(X)+0.2, xticklabels=['0-0', '0-1', '1-0', '1-1'])# 02--4, 12--5, 20--6, 21--7, 22--8
    plt.setp(axarr[1, :], xticks=pd.np.asarray(Xr) + 0.2, xticklabels=['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'])


    #plt.xticks(xticklabels=['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'], rotation=90)
    #plt.setp([a.get_xticklabels() for a in axarr], )
    axarr[1,0].set_xticklabels(['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'], rotation=ang)
    axarr[1,1].set_xticklabels(['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'], rotation=ang)


    axarr[0,0].set_xticklabels(['0-0', '0-1', '1-0', '1-1'], rotation=ang)
    axarr[0,1].set_xticklabels(['0-0', '0-1', '1-0', '1-1'], rotation=ang)



    plt.setp(axarr[1, :], yticks=[1000, 10000], yticklabels=[r'$1\mathrm{e}^3$', r'$1\mathrm{e}^4$'])
    plt.setp(axarr[0, :], yticks=[0,200000,400000, 600000], yticklabels=['0', r'$2\mathrm{e}^5$', r'$4\mathrm{e}^5$', r'$6\mathrm{e}^5$'])

    fig.text(0.01, 0.5, 'Number of Recommendations',va='center', rotation='vertical')
    fig.text(0.5, 0.025, 'Groups', ha='center')
    plt.figlegend(loc='upper center', ncol=2)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.32, wspace=0.05, right=0.99, top=0.87, bottom=0.13)
    plt.savefig("ER.pdf", format='pdf')
    plt.show()



def plotcombiSP(la_fair_f, lon_fair_f, la_reg_f,  lon_reg_f,la_fair_f_race, lon_fair_f_race, la_reg_f_race,  lon_reg_f_race):

    X = pd.np.arange(4)
    fig, axarr = plt.subplots(2, 2,figsize=(fig_width,fig_width/15.*12))
    axarr[0, 0].bar(X, la_reg_f, color='#e66101', alpha=alpha, width=0.40, label='regular')
    axarr[0, 0].bar(X + 0.40, la_fair_f, color='#5e3c99', alpha=alpha, width=0.40, label='fair')
    axarr[0, 0].set_title('LA - gender', {'fontsize': font_title_size})
    axarr[0, 1].bar(X, lon_reg_f, color='#e66101',alpha=alpha,  width=0.40)
    axarr[0, 1].bar(X + 0.40, lon_fair_f, color='#5e3c99', alpha=alpha, width=0.40)
    axarr[0, 1].set_title('London - gender', {'fontsize': font_title_size})

    ### RACE ###

    Xr = pd.np.arange(9)
    axarr[1, 0].bar(Xr, la_reg_f_race, color='#e66101', alpha=alpha, width=0.30)
    axarr[1, 0].bar(Xr + 0.30, la_fair_f_race, color='#5e3c99', alpha=alpha, width=0.30)
    axarr[1, 0].set_title('LA - race', {'fontsize': font_title_size})
    axarr[1, 1].bar(Xr, lon_reg_f_race, color='#e66101', alpha=alpha, width=0.30)
    axarr[1, 1].bar(Xr + 0.30, lon_fair_f_race, color='#5e3c99', alpha=alpha, width=0.30)
    axarr[1, 1].set_title('London - race', {'fontsize': font_title_size})

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    print [ np.format_float_scientific(float(str(l)), precision=1) for l in axarr[1, 1].get_yticks()]
    plt.setp(axarr[1, 0], yticks= axarr[1, 0].get_yticks(),  yticklabels=['0', r'$1\mathrm{e}^{-5}$', r'$2\mathrm{e}^{-5}$', r'$3\mathrm{e}^{-5}$', r'$4\mathrm{e}^{-5}$', '5\mathrm{e}^{-5}$'])
    plt.setp(axarr[0, 0], yticks= axarr[0, 0].get_yticks(),  yticklabels=['0', r'$1\mathrm{e}^{-4}$', r'$2\mathrm{e}^{-4}$', r'$3\mathrm{e}^{-4}$', r'$4\mathrm{e}^{-4}$'])


    plt.setp(axarr[0,:], xticks=pd.np.asarray(X)+0.15, xticklabels=['0-0', '0-1', '1-0', '1-1'])# 02--4, 12--5, 20--6, 21--7, 22--8
    plt.setp(axarr[1, :], xticks=pd.np.asarray(Xr) + 0.15, xticklabels=['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'])


    axarr[1,0].set_xticklabels(['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'], rotation=ang)
    axarr[1,1].set_xticklabels(['0-0', '0-1', '1-0', '1-1', '0-2', '1-2', '2-0', '2-1', '2-2'], rotation=ang)


    axarr[0,0].set_xticklabels(['0-0', '0-1', '1-0', '1-1'], rotation=ang)
    axarr[0,1].set_xticklabels(['0-0', '0-1', '1-0', '1-1'], rotation=ang)


    fig.text(0.01, 0.5, 'Acceptance rates',va='center', rotation='vertical')
    fig.text(0.5, 0.025, 'Groups', ha='center')
    plt.figlegend(loc='upper center', ncol=2)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.32, wspace=0.05, right=0.99, top=0.87, bottom=0.13)

    plt.savefig("SP.pdf", format='pdf')
    plt.show()

from graph_utils import *

def plotGengroups_org(org, reg, fair):
    """
    gender 0.1
    stats = {
     'groups_reg': [5528.4, 2364.0, 2344.8,2976.8 ],
     'groups_fair': [5197.8, 2678.2,  2598.8, 2739.2],
    }
    """

    stats = {'groups_org': org,
             'groups_reg': reg,
             'groups_fair': fair,
             }

    X = pd.np.arange(4)
    plt.bar(X , stats['groups_org'], color='b', width=0.30, label='organic')
    plt.bar(X+0.3, stats['groups_reg'], color='r', width=0.30, label='regular')
    plt.bar(X + 0.60, stats['groups_fair'], color='g', width=0.30, label='fair')

    plt.xticks(map(lambda x: x + 0.45, X), X)
    plt.xlabel('Gender groups',)
    plt.ylabel('Recommended pairs')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotGengroups(la_reg, la_fair, lon_reg, lon_fair, la_reg_f, la_fair_f, lon_reg_f, lon_fair_f):
    """
    gender 0.1
    stats = {
     'groups_reg': [5528.4, 2364.0, 2344.8,2976.8 ],
     'groups_fair': [5197.8, 2678.2,  2598.8, 2739.2],
    }
    """


    X = pd.np.arange(4)
    fig, axarr = plt.subplots(2, 2,figsize=(8,8))
    axarr[0, 0].bar(X, la_reg, color='r', width=0.40, label='regular')
    axarr[0, 0].bar(X + 0.40, la_fair, color='g', width=0.40, label='fair')
    axarr[0, 0].set_title('LA')
    axarr[0, 1].bar(X, lon_reg, color='r', width=0.40)
    axarr[0, 1].bar(X + 0.40, lon_fair, color='g', width=0.40)
    axarr[0, 1].set_title('London')


    axarr[1, 0].bar(X, la_reg_f, color='r', width=0.40)
    axarr[1, 0].bar(X + 0.40, la_fair_f, color='g', width=0.40)

    axarr[1, 1].bar(X, lon_reg_f, color='r', width=0.40)
    axarr[1, 1].bar(X + 0.40, lon_fair_f, color='g', width=0.40)

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    #axarr[0, 0].set_ylabel({'Distribution of';'recommendations';'across groups'})
    axarr[1, 0].set_ylabel('Recommendations as fraction of all possible pairs in each group')
    plt.setp(axarr, xticks=pd.np.asarray(X)+0.2, xticklabels=['0-0', '0-1', '1-0', '1-1'])
    #plt.xticks(map(lambda x: x + 0.02, X), X)
    fig.text(0.5, 0, 'Gender groups', ha='center')
    #plt.xlabel('Gender groups')
    #fig.text(0, 0.5, 'Recommended pairs',va='center', rotation='vertical')

    plt.figlegend(loc='upper center')
    plt.tight_layout()
    plt.show()



def plotRacegroups_org(org, reg, fair):
    """
    race 0.05
    'groups_reg': [325.6
                       , 657.2
                       , 658.6
                       , 4507.8
                       , 36.8
                       , 175.6
                       ,36.2
                       , 171.6
                      , 37.6],
        'groups_fair': [ 429.8, 822.0,851.0,3893.4, 62.6,226.8, 61.6, 214.6,45.2],
    """

    stats = {'groups_org': org,
             'groups_reg': reg,
     'groups_fair': fair,
            }

    X = pd.np.arange(9)
    plt.bar(X, stats['groups_org'], color='b', width=0.30, label='organic')
    plt.bar(X+0.3, stats['groups_reg'], color='r', width=0.30, label='regular')
    plt.bar(X + 0.60, stats['groups_fair'], color='g', width=0.30, label='fair')
    plt.xticks(map(lambda x: x + 0.45, X), X)
    plt.xlabel('Race groups')
    plt.ylabel('Recommended pairs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotRacegroups(reg, fair):
    """
    race 0.05
    'groups_reg': [325.6
                       , 657.2
                       , 658.6
                       , 4507.8
                       , 36.8
                       , 175.6
                       ,36.2
                       , 171.6
                      , 37.6],
        'groups_fair': [ 429.8, 822.0,851.0,3893.4, 62.6,226.8, 61.6, 214.6,45.2],
    """

    stats = {'groups_reg': reg,
     'groups_fair': fair,
            }

    X = pd.np.arange(9)
    plt.bar(X, stats['groups_reg'], color='r', width=0.40, label='regular')
    plt.bar(X + 0.40, stats['groups_fair'], color='g', width=0.40, label='fair')
    plt.xticks(map(lambda x: x + 0.01, X), X)
    plt.xlabel('Race groups')
    plt.ylabel('recommended pairs')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotGC(DATAPATH2,  gc_File_organic, gc_File, gc_File_fair, plotfile, ccdfflag):

    fig= plt.figure(figsize=(15, 10))

    org_df = pd.read_csv(DATAPATH2 + gc_File_organic, index_col=0)
    bins = org_df.range.values
    plt.xticks(range(len(bins)), bins)

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File, index_col=0)
        df_arr.append(df)
    reg_av_df = pd.concat(df_arr).groupby(level=0).mean()
    #bins= df.range.values

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File_fair, index_col=0)
        df_arr.append(df)
    fair_av_df = pd.concat(df_arr).groupby(level=0).mean()

    if ccdfflag:
        # Now find the cdf
        cdf = pd.np.cumsum(reg_av_df.frac_0) / float(reg_av_df.frac_0.sum())
        ccdf = 1 - cdf
        plt.plot(list(ccdf), label='reg growth  ccdf ')


        cdf = pd.np.cumsum(org_df.frac_0) / float(org_df.frac_0.sum())
        ccdf = 1 - cdf
        plt.plot(list(ccdf), label='org growth  ccdf ')


        cdf = pd.np.cumsum(fair_av_df.frac_0) / float(fair_av_df.frac_0.sum())
        ccdf = 1 - cdf
        plt.plot(list(ccdf), label='fair growth  ccdf ')

    else:
        #yhat = savgol_filter(av_df_reg.iloc[:,i], 101, 2)
        #axarr[i].plot(yhat, label='Regular growth')
        plt.plot(reg_av_df.frac_0, label='reg growth')
        plt.plot(org_df.frac_0, label='organic growth')
        plt.plot(fair_av_df.frac_0, label='Fair growth')
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(DATAPATH2+'plots/' +plotfile)
    plt.show()


def plotGC_race_together(DATAPATH2,  gc_File_organic, gc_File, gc_File_fair, plotfile, ccdfflag=False):
    fig= plt.figure(figsize=(18, 10))

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File, index_col=0)
        df_arr.append(df)
    av_df_reg = pd.concat(df_arr).groupby(level=0).mean()
    # bins= df.range.values

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File_fair, index_col=0)
        df_arr.append(df)
    av_df_fair = pd.concat(df_arr).groupby(level=0).mean()

    org_df = pd.read_csv(DATAPATH2 + gc_File_organic, index_col=0)
    bins = org_df.range.values

    plt.xticks(range(len(bins)), bins)

    for i in range(3):
        if ccdfflag:
            # Now find the cdf
            cdf = pd.np.cumsum(av_df_fair.iloc[:,i])/float(av_df_fair.iloc[:,i].sum())
            ccdf=1-cdf
            plt.plot( list(ccdf), label='Fair growth Race ccdf ' +str(i))


            """cdf = pd.np.cumsum(av_df_reg.iloc[:,i])/float(av_df_reg.iloc[:,i].sum())
            ccdf=1-cdf
            plt.plot(list(ccdf), label='Reg growth Race ccdf ' +str(i))

            cdf = pd.np.cumsum(org_df.iloc[:, i+1]) / float(org_df.iloc[:, i+1].sum())
            ccdf = 1 - cdf
            plt.plot( list(ccdf), label='Org growth Race ccdf ' + str(i))"""

        elif not ccdfflag:
            plt.plot(av_df_fair.iloc[:,i], label='Fair growth Race '+str(i))

            #plt.plot(av_df_reg.iloc[:,i], label= 'Regular growth Race '+str(i))
            #plt.plot(org_df.iloc[:,i+1], label='organic growth Race '+str(i))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim([0,1])
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(DATAPATH2+'plots/' + plotfile)
    plt.show()


def plotGC_race(DATAPATH2,  gc_File_organic, gc_File, gc_File_fair, plotfile):

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File, index_col=0)
        df_arr.append(df)
    av_df_reg = pd.concat(df_arr).groupby(level=0).mean()
    # bins= df.range.values

    df_arr = []
    for i in range(5):
        DATAPATH = DATAPATH2 + str(i) + "/"
        df = pd.read_csv(DATAPATH + gc_File_fair, index_col=0)
        df_arr.append(df)
    av_df_fair = pd.concat(df_arr).groupby(level=0).mean()

    org_df = pd.read_csv(DATAPATH2 + gc_File_organic, index_col=0)
    bins = org_df.range.values

    f, axarr = plt.subplots(3, sharex=True, figsize=(15, 10))
    plt.xticks(range(len(bins)), bins)

    for i in range(3):
        axarr[i].set_yscale('log')
        axarr[i].set_title('Race '+str(i))



        axarr[i].plot(av_df_reg.iloc[:,i], label= 'Regular growth')
        #yhat = savgol_filter(av_df_reg.iloc[:,i], 101, 2)
        #axarr[i].plot(yhat, label='Regular growth')

        axarr[i].plot(av_df_fair.iloc[:,i], label='Fair growth')
        axarr[i].plot(org_df.iloc[:,i+1], label='organic growth')

    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(DATAPATH2+'plots/' + plotfile)
    plt.show()






def plotROCs(DATAPATH, window_size, polynomial_order, plotfile_pref, rocfile_suff, race):

    plt.figure()
    groups= 9 if race else 4

    for gid in range(groups):
        ROC_df=pd.read_csv(DATAPATH +str(gid)+ rocfile_suff, index_col=0)#../../data/london/"+ str(0)+"_ROC_df.csv"
        df = ROC_df.sort_values(by=['fpr'])
        yhat = savgol_filter(df.tpr, window_size, polynomial_order)
        plt.plot(df.fpr, yhat, label='group_' + str(gid))
        print len(df.tpr)

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (false alarm rate)')
    plt.ylabel('True Positive Rate or (positives are friends)')
    plt.title('Receiver Operating Characteristic Image')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold',color='r')
    # ax2.set_ylim([thresholds[-1],thresholds[1]])
    # ax2.set_xlim([fpr[0],fpr[-1]])
    plt.grid(True)
    print 'finished'
    plt.savefig(DATAPATH+ plotfile_pref+str(window_size) + str(polynomial_order)  +'.png')
    plt.show()


def plot_bias(DATAPATH, fn):

    df = pd.read_csv(DATAPATH+fn, index_col=0)#, skipfooter=1)
    print df.shape
    df=df[df['gender[u]']!=-1]
    df=df[df['race[u]'] != -1]
    df.to_csv(DATAPATH+fn)
    print df.shape
    fig= plt.figure(figsize=(60, 80))
    columns = df.columns[3:]
    #columns=['sup_same_gen_bias100', 'sup_fem_gen_bias100', 'sup_same_race_bias100', 'sup_0_race_bias100', 'sup_1_race_bias100', 'sup_2_race_bias100',
    #'sup_same_gen_bias50', 'sup_fem_gen_bias50', 'sup_same_race_bias50', 'sup_0_race_bias50', 'sup_1_race_bias50', 'sup_2_race_bias50',
    #'sup_same_gen_bias10', 'sup_fem_gen_bias10', 'sup_same_race_bias10', 'sup_0_race_bias10', 'sup_1_race_bias10', 'sup_2_race_bias10']

    # df.set_index("name",drop=True,inplace=True)
    axarr=df.hist(column=columns, bins=20, figsize=(15, 15))
    #for ax in axarr.flatten():
    #    ax.set_xticks(arange(0.0,1.0,0.1))
        #ax.set_yticks(range(0,500, 50))

    plt.legend(fontsize=25)
    plt.xlim([0.0, 1.0])
    #plt.yticks(fontsize=30)`
    plt.savefig(DATAPATH+"plots/"+fn+".png")
    plt.show()


def get_recommendation_statistics(path, equality='geneq', top_frac=0.25, per_usr=10):
    '''
    Get recommendation genders (and later maybe races) and their percentage
    for normal (nor) and gendeq (geq) or raceeq (req) walk (depending on equality).
    For both top `top_frac` fraction of recommendations
    and `per_usr` best recommendations for each user.
    
    Save the results in `path`/recommendation_stats_nor_geq.json
    '''
    if equality == 'geneq':
        groups = {0: 0, 1: 1, 2: 0, 3: 1}
    elif equality == 'raceeq':
        groups = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 0, 7: 1, 8: 2}
    else:
        raise Exception("equality should be either 'geneq' or 'raceeq'")
    
    recom_top_nor = {}
    recom_top_eq = {}
    recom_pusr_nor = {}
    recom_pusr_eq = {}
    for i in range(5):
        rec_nor = pd.read_csv('{}/{}/recommendations.csv'.format(path, i), index_col=0)
        rec_eq = pd.read_csv('{}/{}/recommendations_{}.csv'.format(path, i, equality), index_col=0)
        # rec_nor.indices_desc = [int(i) for i in rec_nor.indices_desc]
        # rec_geq.indices_desc = [int(i) for i in rec_geq.indices_desc]
        rec_nor_srtd = rec_nor.sort_values(by=['probs'], ascending=False)
        rec_eq_srtd = rec_eq.sort_values(by=['probs'], ascending=False)
        hada_nor = pd.read_csv('{}/{}/hada_100_test.csv'.format(path, i),
                               header=None, error_bad_lines=False,
                               names=['u1', 'u2', 'label', 'group']+list(range(128)))
        hada_eq = pd.read_csv('{}/{}/hada_100_test_{}.csv'.format(path, i, equality),
                               header=None, error_bad_lines=False,
                               names=['u1', 'u2', 'label', 'group']+list(range(128)))
        
        # Get 'top_frac' best recommendations of all.
        for u1, group in zip(
                hada_nor.u1[rec_nor_srtd.head(int(len(rec_nor) * top_frac)).indices_desc],
                hada_nor.group[rec_nor_srtd.head(int(len(rec_nor) * top_frac)).indices_desc]):
            if u1 in recom_top_nor:
                recom_top_nor[u1].append(groups[group])
            else:
                recom_top_nor[u1] = [groups[group]]
        for u1, group in zip(
                hada_eq.u1[rec_eq_srtd.head(int(len(rec_nor) * top_frac)).indices_desc],
                hada_eq.group[rec_eq_srtd.head(int(len(rec_nor) * top_frac)).indices_desc]
                ):
            if u1 in recom_top_eq:
                recom_top_eq[u1].append(groups[group])
            else:
                recom_top_eq[u1] = [groups[group]]
        # Get `per_usr` best recommendations for each user.
        for u1, group in zip(
                hada_nor.u1[rec_nor_srtd.indices_desc],
                hada_nor.group[rec_nor_srtd.indices_desc]
                ):
            if u1 not in recom_pusr_nor:
                recom_pusr_nor[u1] = [groups[group]]
            elif len(recom_pusr_nor[u1]) < per_usr:
                recom_pusr_nor[u1].append(groups[group])
        for u1, group in zip(
                hada_eq.u1[rec_eq_srtd.indices_desc],
                hada_eq.group[rec_eq_srtd.indices_desc]
                ):
            if u1 not in recom_pusr_eq:
                recom_pusr_eq[u1] = [groups[group]]
            elif len(recom_pusr_eq[u1]) < per_usr:
                recom_pusr_eq[u1].append(groups[group])
                
    # calculate percentage of recommendations being 0, 1, 2 for raceeq or just 0 for geneq
    number_of_groups = 3 if equality == 'raceeq' else 1
    for i in range(number_of_groups):
        stats_top_nor = [rec.count(i) / float(len(rec)) for rec in recom_top_nor.itervalues()]
        stats_top_eq = [rec.count(i) / float(len(rec)) for rec in recom_top_eq.itervalues()]
        stats_pusr_nor = [rec.count(i) / float(len(rec)) for rec in recom_pusr_nor.itervalues()]
        stats_pusr_eq = [rec.count(i) / float(len(rec)) for rec in recom_pusr_eq.itervalues()]
        open('{}/recommendation_stats_nor_{}_{}.json'.format(path, equality, i), 'w').write(
                json.dumps([stats_top_nor, stats_top_eq, stats_pusr_nor, stats_pusr_eq]))
        
def get_statistics_from_recommendation(
        path, racegen_name,
        recom_name_normal='topRecos.edgelist',
        recom_name_race='topRecos_raceeq.edgelist',
        recom_name_gend='topRecos_geneq.edgelist',
        fraction=1.,
        drop=5):

    '''
    Get recommendation genders and races and their percentage
    for normal (nor) and eqal (raceeq, gendeq) walk.
    Take `fraction` highest scoring edges.
    Drop all nodes that have len(recommended) <= `drop`
    
    Save the results in
        `path`/recom_stats_gendeq_0.json
        `path`/recom_stats_raceeq_0.json
        `path`/recom_stats_raceeq_1.json
        `path`/recom_stats_raceeq_2.json
    '''
    if not recom_name_race and not recom_name_gend:
        raise Exception("give me at least something to work with!")
    
    recom_top_gnor = {}
    recom_top_rnor = {}
    recom_top_geq = {}
    recom_top_req = {}
    for i in range(5):
        rec_nor = pd.read_csv('{}/{}/{}'.format(path, i, recom_name_normal))
        rec_nor = rec_nor.tail(int(len(rec_nor) * fraction))
        if recom_name_race:
            rec_race = pd.read_csv('{}/{}/{}'.format(path, i, recom_name_race))
            rec_race = rec_race.tail(int(len(rec_race) * fraction))
        if recom_name_gend:
            rec_gend = pd.read_csv('{}/{}/{}'.format(path, i, recom_name_gend))
            rec_gend = rec_gend.tail(int(len(rec_race) * fraction))
        gender, race = read_instagram_genrace('{}/{}'.format(path, racegen_name))
        
        for _, u, v, _ in rec_nor.itertuples():
            recom_top_gnor[u] = recom_top_gnor.get(u, []) + [gender[v]]
            recom_top_rnor[u] = recom_top_rnor.get(u, []) + [race[v]]
        
        if recom_name_gend:
            for _, u, v, _ in rec_gend.itertuples():
                recom_top_geq[u] = recom_top_geq.get(u, []) + [gender[v]]
                
        if recom_name_race:
            for _, u, v, _ in rec_race.itertuples():
                recom_top_req[u] = recom_top_req.get(u, []) + [race[v]]
                
    # calculate percentage of recommendations being 0, 1, 2 for raceeq or just 0 for geneq
    if recom_name_race:
        for i in range(3):
            stats_top_nor = [rec.count(i) / float(len(rec)) for rec in recom_top_rnor.values() if len(rec) > drop]
            stats_top_eq = [rec.count(i) / float(len(rec)) for rec in recom_top_req.values() if len(rec) > drop]
            open('{}/recom_stats_raceeq_{}_{}.json'.format(path, i, fraction), 'w').write(
                    json.dumps([stats_top_nor, stats_top_eq]))
    if recom_name_gend:
        stats_top_nor = [rec.count(0) / float(len(rec)) for rec in recom_top_gnor.values() if len(rec) > drop]
        stats_top_eq = [rec.count(0) / float(len(rec)) for rec in recom_top_geq.values() if len(rec) > drop]
        open('{}/recom_stats_gendeq_{}_{}.json'.format(path, 0, fraction), 'w').write(
                json.dumps([stats_top_nor, stats_top_eq]))


def get_statistics_from_friendships(path, genrace_file, drop=5):
    '''
    Get statistics in the same format as get_statistics_from_recommendations.

    :param path: path to folder with 0,1,2,3,4 folders e.g. ../data/la
    :param genrace_file: path to genrace file e.g. ../data/la/la.genrace
    :param drop: all nodes with less then `drop` friends will be dropped from statistics
    :return: will generate json files in path/frien_stats_(...)
    '''
    gender, race = read_instagram_genrace(genrace_file)
    fr_gen = {}
    fr_race = {}
    for i in range(5):
        graph = read_nxgraph('{}/{}/known_80.edgelist'.format(path, i))
        for u in graph.nodes():
            friends = list(graph.neighbors(u))
            fr_gen[u] = fr_gen.get(u, []) + [gender[friend] for friend in friends]
            fr_race[u] = fr_race.get(u, []) + [race[friend] for friend in friends]
    for i in range(3):
        stats_fr = [fr.count(i) / float(len(fr)) for fr in fr_race.values() if len(fr) > drop]
        write_json(stats_fr, '{}/frien_stats_race_{}.json'.format(path, i))
    stats_fr = [fr.count(0) / float(len(fr)) for fr in fr_gen.values() if len(fr) > drop]
    write_json(stats_fr, '{}/frien_stats_gend_{}.json'.format(path, 0))


def plot_recommendation_statistics(path, filename='recom_stats', original='frien_stats'):
    
    bins = 30
    fig, axarr = plt.subplots(1, 4,figsize=(8,4), sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.75, wspace=0.06, hspace=0.15)
    fig.text(0.5, 0.04, 'fraction of protected group', ha='center')
    fig.text(0.02, 0.5, 'number of users', va='center', rotation='vertical')

    subplot = axarr[0]
    subplot.set_yticklabels([])
    stats_top_nor, stats_top_geq = json.loads(
            open('{}/{}_gendeq_0.json'.format(path, filename), 'r').read())

    if original:
        stats_ori = read_json('{}/{}_gend_0.json'.format(path, original))
    
    _, _, reg_patches = subplot.hist(stats_top_nor, bins, alpha=0.5, label='regular')
    _, _, fair_patches = subplot.hist(stats_top_geq, bins, alpha=0.5, label='fair')
    if original:
        _, _, ori_patches = subplot.hist(stats_ori, bins, alpha=0.5, label='friends')
    subplot.axvline(x=0.5)

    #subplot.legend(loc='upper right')
    subplot.set_title('gender 0', {'fontsize': font_title_size})
    print('gender 0')
    if original:
        print(' original mean: {:.3f} std: {:.3f}'.format(
            abs(.5 - np.mean(stats_ori)), np.std(stats_ori)))
    print(' normal mean: {:.3f} std: {:.3f}\n fair mean: {:.3f} std: {:.3f}'.format(
           abs(.5 - np.mean(stats_top_nor)), np.std(stats_top_nor),
           abs(.5 - np.mean(stats_top_geq)), np.std(stats_top_geq)))
    print('gender 1')
    if original:
        print(' original mean: {:.3f} std: {:.3f}'.format(
            np.mean(stats_ori) - .5, np.std(stats_ori)))
    print(' normal mean: {:.3f} std: {:.3f}\n fair mean: {:.3f} std: {:.3f}'.format(
           np.mean(stats_top_nor) - .5, np.std(stats_top_nor),
           np.mean(stats_top_geq) - .5, np.std(stats_top_geq)))


    for i in range(3):
        stats_top_nor, stats_top_geq = json.loads(
                open('{}/{}_raceeq_{}.json'.format(path, filename, i), 'r').read())
        if original:
            stats_ori = read_json('{}/{}_race_{}.json'.format(path, original, i))
        
        subplot = axarr[i+1]
        subplot.set_yticklabels([])
        subplot.hist(stats_top_nor, bins, alpha=0.5, label='regular')
        subplot.hist(stats_top_geq, bins, alpha=0.5, label='fair')
        if original:
            subplot.hist(stats_ori, bins, alpha=0.5, label='friends')
        #subplot.legend(loc='upper left' if i == 1 else 'upper right')
        subplot.axvline(x=1./3)
        subplot.set_title('race {}'.format(i), {'fontsize': font_title_size})
        print('race {}'.format(i))
        if original:
            print(' original mean: {:.3f} std: {:.3f}'.format(
               abs(np.mean(stats_ori) - 1./3), np.std(stats_ori)))
        print(' normal mean: {:.3f} std: {:.3f}\n gendeq mean: {:.3f} std: {:.3f}'.format(
               abs(np.mean(stats_top_nor) - 1./3), np.std(stats_top_nor),
               abs(np.mean(stats_top_geq) - 1./3), np.std(stats_top_geq)))

    if original:
        plt.figlegend((reg_patches[0], fair_patches[0], ori_patches[0]),
                      ('regular', 'fair', 'friends'), loc='upper center',
                      ncol=3)
    else:
        plt.figlegend((reg_patches[0], fair_patches[0]),
                      ('regular', 'fair'), loc='upper center',
                      ncol=2)
    plt.savefig('../paper_IJCAI/graphics/la_histograms.pdf', format='pdf')
    plt.show()


def plot_recommendation_statistics_lon_la(lon_path='../data/london', la_path='../data/la', filename='recom_stats',
                                   original='frien_stats'):
    set_font()
    plt.rcParams['text.usetex'] = True
    font_title_size = 34
    font_box_size = 22
    bins = 30
    fig, axarr = plt.subplots(2, 4, figsize=(fig_width, fig_width/4*3), sharex=False)
    #alpha = 0.7
    #color_network = '#b2abd2'
    #color_standard = '#e66101'
    #color_fair = '#5e3c99'
    colors = [color_standard, color_fair]
    hatches = ['', '']
    labels = ['regular', 'fair']
    if original:
        colors.insert(0, color_network)
        labels.insert(0, 'network')

    left = 0.09
    right = 0.98
    bottom = 0.13
    top = 0.825
    wspace = 0.06
    hspace = 0.05
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    fig.text(left + (right - left) / 2, 0.04, 'fraction of protected group', ha='center')
    fig.text(0.06, bottom + (top - bottom) / 2, 'number of users', va='center', rotation='vertical')
    fig.text(0.02, (top-bottom-hspace)/4+bottom, 'London', va='center', rotation='vertical', fontsize=font_title_size)
    fig.text(0.02, (top-bottom-hspace)*3/4+hspace+bottom, 'Los Angeles', va='center', rotation='vertical', fontsize=font_title_size)

    def one_subplot(subplot, stats, labels, colors, title, row):
        fair_mean = 0.5 if title.startswith('gend') else 1./3
        subplot.set_yticklabels([])
        subplot.set_xticks([0, fair_mean, 1])
        if row == 1:
            subplot.set_xticklabels(['0', '{:.1f}'.format(fair_mean), '1'])
        else:
            subplot.set_xticklabels([])
        patches = [
            subplot.hist(stat, bins, range=(0,1), alpha=alpha, label=label, color=color, hatch=hatch)[2]
            for (stat, label, color, hatch) in zip(stats, labels, colors, hatches)]
        subplot.axvline(x=fair_mean, color='black', linestyle='--')
        if row == 0:
            subplot.set_title(title, {'fontsize': font_title_size})

        #value printing
        print(title)
        for stat, label in zip(stats, labels):
            print('  {} mean: {:.3f} std: {:.3f}'.format(
                label, fair_mean - np.mean(stat), np.std(stat)
            ))
        if title.startswith('gend'):
            print('gender 1')
            for stat, label in zip(stats, labels):
                print('  {} bias: {:.3f} std: {:.3f}'.format(
                    label, fair_mean - (1 - np.mean(stat)), np.std(stat)
                ))
        return patches

    for row, path in enumerate([la_path, lon_path]):
        subplot = axarr[row, 0]
        stats = read_json('{}/{}_gendeq_0.json'.format(path, filename))
        if original:
            stats.insert(0, read_json('{}/{}_gend_0.json'.format(path, original)))

        patches = one_subplot(subplot, stats, labels, colors, 'gender 0', row)

        for i in range(3):
            stats = read_json('{}/{}_raceeq_{}.json'.format(path, filename, i))
            if original:
                stats.insert(0, read_json('{}/{}_race_{}.json'.format(path, original, i)))
            subplot = axarr[row, i + 1]
            one_subplot(subplot, stats, labels, colors, 'race {}'.format(i), row)

    if original:
        plt.figlegend((patches[0][0], patches[1][0], patches[2][0]),
                      ('network', 'regular', 'fair'), loc='upper center',
                      ncol=3)
    else:
        plt.figlegend((patches[0][0], patches[1][0]),
                      ('regular', 'fair'), loc='upper center',
                      ncol=2)
    plt.savefig('../paper_IJCAI/graphics/la_lon_histograms.pdf', format='pdf')
    plt.show()

def get_counters_gender_race_walk_percentage(path, genrace_file):
    def get_nodes_from_walk(walk_file):
        nodes_lines = [line.strip() for line in open(walk_file, 'r').readlines()]
        return [int(nd) for nd in ','.join(nodes_lines).split(',')]
    
    gender, race = read_instagram_genrace(genrace_file)
    counters = {
            'net': {'gend': [0, 0], 'race': [0, 0, 0]},
            '': {'gend': [0, 0], 'race': [0, 0, 0]},
            '_gendeq': {'gend': [0, 0], 'race': [0, 0, 0]},
            '_raceeq': {'gend': [0, 0], 'race': [0, 0, 0]},
            }
    for i in range(5):        
        print('starting {}'.format(i))
        for walk_type in ('net', '', '_gendeq', '_raceeq'):
            print('\tstarting {} type'.format(walk_type))
            if walk_type == 'net':
                nodes = read_nxgraph('{}/{}/known_80.edgelist'.format(path, i)).nodes()
            else:
                nodes = get_nodes_from_walk('{}/{}/known_80{}.walk'.format(path, i, walk_type))
            for node in nodes:
                counters[walk_type]['gend'][gender[node]] += 1
                counters[walk_type]['race'][race[node]] += 1
    print('writing pickle')
    write_pickle(counters, '{}/counters.pick'.format(path))
    
    
def plot_gender_race_walk_percentage(counters_pick):
    plt.rcParams['text.usetex'] = True
    set_font()
    counters = read_pickle(counters_pick)
    stats = {'ogc': [float(val) / sum(counters['net']['gend']) for val in counters['net']['gend']],
             'orc': [float(val) / sum(counters['net']['race']) for val in counters['net']['race']],
             'wegc': [float(val) / sum(counters['_gendeq']['gend']) for val in counters['_gendeq']['gend']],
             'werc': [float(val) / sum(counters['_raceeq']['race']) for val in counters['_raceeq']['race']],
             'wgc': [float(val) / sum(counters['']['gend']) for val in counters['']['gend']],
             'wrc': [float(val) / sum(counters['']['race']) for val in counters['']['race']]}

    #color_network = '#b2abd2'
    #color_standard = '#e66101'
    #color_fair = '#5e3c99'

    fig, axarr = plt.subplots(1, 2,figsize=(fig_width,fig_width/3.), sharey=True)
    plt.subplots_adjust(bottom=0.2, top=0.8, left = 0.1, right=0.98, wspace=0.05)
    subplot = axarr[0]
    
    X = np.arange(2)
    network_patches = subplot.bar(X + 0.00, stats['ogc'],alpha = alpha, color = color_network, width = 0.25, label = 'network')
    regular_patches = subplot.bar(X + 0.25, stats['wgc'],alpha = alpha, color = color_standard, width = 0.25, label = 'regular')
    fair_patches = subplot.bar(X + 0.50, stats['wegc'], alpha = alpha, color = color_fair, width = 0.25, label = 'fair')
    subplot.set_xticks([0.25, 1.25])
    subplot.set_xticklabels(['0', '1'])
    subplot.set_xlabel('gender')
    subplot.set_ylabel('percentage')
    #subplot.legend()
    
    #plt.show()
    
    subplot = axarr[1]
    X = np.arange(3)
    subplot.bar(X + 0.00, stats['orc'], alpha = alpha, color = color_network, width = 0.25, label = 'network')
    subplot.bar(X + 0.25, stats['wrc'], alpha = alpha, color = color_standard, width = 0.25, label = 'regular')
    subplot.bar(X + 0.50, stats['werc'], alpha = alpha, color = color_fair, width = 0.25, label = 'fair')
    subplot.set_xticks([0.25, 1.25, 2.25])
    subplot.set_xticklabels(['0', '1', '2'])
    subplot.set_xlabel('race')
    #subplot.set_ylabel('percentage')
    #subplot.legend()

    plt.figlegend((network_patches[0], regular_patches[0], fair_patches[0]),
                  ('network', 'regular', 'fair'), loc='upper center',
                  ncol=3)
    plt.savefig('../paper_IJCAI/graphics/la_walk_percentages.pdf', format='pdf')
    plt.show()
    
    
def node2vec_evaluation_plot():
    plt.rcParams['text.usetex'] = True
    filename = 'recom_stats'
    original = 'frien_stats'

    #color_network = '#b2abd2'
    #color_standard = '#e66101'
    #color_fair = '#5e3c99'
    
    hatches = ['/', '\\']
    set_font()
    font_title_size = 34
    font_box_size = 22
    bins = 30
    fig, axarr = plt.subplots(2, 3, figsize=(fig_width, fig_width/3.*2), sharex=False)
    left = 0.09
    right = 0.98
    wspace=0.06
    bottom = 0.13
    top = 0.835
    hspace=0.05
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    fig.text(0.02, (top-bottom-hspace)/4+bottom, 'London', va='center', rotation='vertical', fontsize=font_title_size)
    fig.text(0.02, (top-bottom-hspace)*3/4+hspace+bottom, 'Los Angeles', va='center', rotation='vertical', fontsize=font_title_size)
    fig.text(left + (right - left)/2, 0.04, 'fraction of protected group', ha='center')
    fig.text(0.06, bottom + (top - bottom)/2, 'number of users', va='center', rotation='vertical')

    for row, path in enumerate(['../data/la', '../data/london']):
        sub_num = 0
        subplot = axarr[row, sub_num]
        subplot.set_yticklabels([])
        subplot.set_xticks([0, 0.5, 1])
        if row == 1:
            subplot.set_xticklabels(['0', '0.5', '1'])
        else:
            subplot.set_xticklabels([])
        subplot.axvline(x=0.5, color='black', linestyle='--')
        stats_top_nor, stats_top_geq = read_json('{}/{}_gendeq_0.json'.format(path, filename))
        stats_top_nor_1 = [1 - i for i in stats_top_nor]
        if original:
            stats_ori = read_json('{}/{}_gend_0.json'.format(path, original))
            stats_ori_1 = [1 - i for i in stats_ori]

        _, _, reg_patches = subplot.hist(stats_top_nor_1, bins, range=(0,1), alpha=alpha, label='node2vec', color=color_standard, hatch='\\')
        if original:
            _, _, ori_patches = subplot.hist(stats_ori_1, bins, range=(0,1), alpha=alpha, label='network', color=color_network, hatch='/')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text_str = r'bias$^{\mathtt{ERu}}$'+' of\n network:\n  {:.3f}\n node2vec:\n  {:.3f}'.format(
            .5 - np.mean(stats_ori_1),
            .5 - np.mean(stats_top_nor_1))
        subplot.text(
            0.57, 0.95, text_str, verticalalignment='top', horizontalalignment='left',
            bbox=props, transform=subplot.transAxes, fontsize=font_box_size)

        # subplot.legend(loc='upper right')
        if row == 0:
            subplot.set_title('gender 1', {'fontsize': font_title_size})
        print('normal mean: {} std: {}'.format(
            np.mean(stats_top_nor_1), np.std(stats_top_nor_1)))
        if original:
            print('original mean: {} std: {}'.format(
                np.mean(stats_ori_1), np.std(stats_ori_1)))

        for i in [0, 2]:
            stats_top_nor, stats_top_geq = json.loads(
                open('{}/{}_raceeq_{}.json'.format(path, filename, i), 'r').read())
            if original:
                stats_ori = read_json('{}/{}_race_{}.json'.format(path, original, i))
            sub_num += 1
            subplot = axarr[row, sub_num]
            subplot.set_yticklabels([])
            subplot.set_xticks([0, 1./3, 1])
            if row == 1:
                subplot.set_xticklabels(['0', '0.3', '1'])
            else:
                subplot.set_xticklabels([])
            subplot.axvline(x=1./3, color='black', linestyle='--')
            subplot.hist(stats_top_nor, bins, range=(0,1), alpha=alpha, label='node2vec', color=color_standard, hatch='\\')
            if original:
                subplot.hist(stats_ori, bins, range=(0,1), alpha=alpha, label='network', color=color_network, hatch='/')
            # subplot.legend(loc='upper left' if i == 1 else 'upper right')
            if row == 0:
                subplot.set_title('race {}'.format(i), {'fontsize': font_title_size})

            text_str = r'bias$^{\mathtt{ERu}}$'+' of\n network:\n  {:.3f}\n node2vec:\n  {:.3f}'.format(
                .5 - np.mean(stats_ori),
                .5 - np.mean(stats_top_nor))
            subplot.text(
                0.57, 0.95, text_str, verticalalignment='top', horizontalalignment='left',
                bbox=props, transform=subplot.transAxes, fontsize=font_box_size)

            print('normal mean: {} std: {}'.format(
                np.mean(stats_top_nor), np.std(stats_top_nor)))
            if original:
                print('original mean: {} std: {}'.format(
                    np.mean(stats_ori), np.std(stats_ori)))

        if original:
            plt.figlegend((reg_patches[0], ori_patches[0]),
                          ('node2vec', 'network'), loc='upper center',
                          ncol=2)

    plt.savefig('../paper_IJCAI/graphics/n2v_friends_hist.pdf', format='pdf')
    plt.show()
    