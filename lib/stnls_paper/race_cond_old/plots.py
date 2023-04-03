
"""
This graphic shows the errors incurred by the race condition

"""


# -- misc --
import copy,os,random
dcopy = copy.deepcopy
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- data mng --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- vision --
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF

# -- data io --
import data_hub

# -- caching --
import cache_io

# -- management --
from pathlib import Path
from easydict import EasyDict as edict

# -- results packages --
import stnls
from stnls.utils.misc import rslice,read_pickle,write_pickle
from stnls.utils.timer import ExpTimer
from stnls.utils.inds import get_nums_hw
#get_batching_info


# -- plotting --
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# -- other --
SAVE_DIR = Path("./output/race_cond/")

def filter_records(records,filters):
    for key,val in filters.items():
        records = records[records[key] == val]
    return records

def run(records):
    records = records[records['ngroups'].isin([-1,0,1,3,5])]
    error_and_time(records)
    # errors_vs_channels(records)
    # time_vs_channels(records)

def error_and_time(records):
    # -- only nbwd == 1 --
    # records = records[records["nbwd"].isin([1,15])]
    records = records.reset_index(drop=True)

    # -- make times log values --
    idx = records.index
    data = records.loc[:,'dtime'].astype(np.float32)
    # records.loc[idx,'dtime'] = np.log10(data)
    records.loc[idx,'dtime'] = np.log10(data)
    data = records.loc[:,'exact_time'].astype(np.float32)
    records.loc[idx,'exact_time'] = np.log10(data)

    # -- split data --
    edf = records[records['exact'] == True]
    # print(edf['dtime'])
    records = records[records['exact'] == False]
    # print(records[['dtime','exact_time','nbwd','nchnls','rbwd']])

    # -- order data --
    r0 = records[records['ngroups'] == 0]
    rg0 = records[records['ngroups'] > 0]
    records = rg0.append(r0)

    #
    # -- two side-by-side plots --
    #

    # -- plot constants --
    FSIZE = 12
    FSIZE_B = 14
    FSIZE_S = 10

    # -- init plot --
    ginfo = {'width_ratios': [0.35,.35],'wspace':0.20, 'hspace':0.0,
             "top":0.86,"bottom":0.16,"left":0.09,"right":0.99}
    # ginfo = {'width_ratios': [0.47,.47],'wspace':0.05, 'hspace':0.0,
    #          "top":0.80,"bottom":0.16,"left":0.115,"right":0.99}
    fig,ax = plt.subplots(1,2,figsize=(8,4),gridspec_kw=ginfo)

    # -- colors to nbwd --
    cmap = LinearSegmentedColormap.from_list("",["blue","yellow"])
    nunique_nbwd = len(records['ngroups'].unique())
    markers = ['+', 'x', '*', '.', '^','s']
    markers = [v for v in markers for _ in range(2)]
    ax[0].set_prop_cycle(marker=markers)
    ax[1].set_prop_cycle(marker=markers)

    # -- create data to plot --
    handles = []
    h_labels = []
    b = 0
    fields = ['dtime','errors_m']
    for nbwd,bdf in records.groupby("ngroups",sort=False):

        # -- unpack --
        ls = {True:"-",False:"-."}

        # -- plot --
        bdf = bdf.sort_values("rbwd")
        for i,field in enumerate(fields):

            # -- color --
            bwd_s = nbwd if i == 0 else None
            if bwd_s == 0: bwd_s = "C"
            col = b/(float(nunique_nbwd)-1)
            # print(col)
            color = cmap(col)
            for rbwd,bdf_rbwd in bdf.groupby("rbwd"):
                # print(rbwd,ls[rbwd])
                yvals = bdf_rbwd[field]
                xvals = bdf_rbwd['nchnls']
                if rbwd == False: label = ""
                else: label = bwd_s
                h = ax[i].plot(xvals, yvals, color=color,label=label,
                               linewidth=2,linestyle=ls[rbwd],
                               markersize=10)
                if rbwd == True:
                    if i == 0:
                        handles.append(h)
                        h_labels.append(label)

        # -- update --
        b += 1

    # -- plot exact time --
    for i in range(1):
        xvals = edf['nchnls'].to_numpy()
        yvals_gpu = edf['dtime'].to_numpy()
        yvals_cpu = edf['exact_time'].to_numpy()
        order = np.argsort(xvals)
        xvals = xvals[order]
        yvals_cpu = yvals_cpu[order]
        yvals_gpu = yvals_gpu[order]
        label = "%s" % "CPU" if i == 0 else ""
        ax[i].plot(xvals,yvals_cpu,color='k',#label=label,
                   markersize=10,linewidth=2)
        label = "%s" % "GPU" if i == 0 else ""
        ax[i].plot(xvals,yvals_gpu,color='k',#label=label,
                   markersize=10,linewidth=2)

    # -- reset ticks --
    ylabels = ["Log10 Wall-Clock Time (sec)","Relative Error"]
    for i,field in enumerate(fields):
        y_ours = records[field]
        if field == "dtime": y_exact = edf['dtime'] # exact cpu time
        else: y_exact = y_ours
        x = records['nchnls'].to_numpy()
        x = np.sort(np.unique(x))
        xmin,xmax = x.min().item(),x.max().item()
        if field == "dtime":
            ymin,ymax = (y_ours*1.1).min().item(),(y_exact*1.1).max().item()
        else:
            ymin,ymax = (y_ours*.8).min().item(),(y_exact*1.05).max().item()
        yticks = np.linspace(ymin,ymax,5)
        yticklabels = ["%1.1f" % x for x in yticks]
        xticks = x#np.linspace(xmin,xmax,4)
        xticklabels = ["%d" % x for x in xticks]
        ax[i].set_yticks(yticks)
        ax[i].set_yticklabels(yticklabels,fontsize=FSIZE)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticklabels,fontsize=FSIZE)
        ax[i].set_ylim(ymin,ymax)
        ax[i].set_ylabel(ylabels[i],fontsize=FSIZE)
    ax[0].set_xlabel("Number of Channels (C)",fontsize=FSIZE)
    ax[1].set_xlabel("Number of Channels (C)",fontsize=FSIZE)

    # -- format titles --
    fig.suptitle("Comparing Execution Times",fontsize=FSIZE_B)
    # fig.suptitle("Fast, Approximate Backpropagation",fontsize=FSIZE_B)
    fig.suptitle("Reducing Errors from Race Condition",fontsize=FSIZE_B)
    ax[0].set_title("Backpropagation Runtime",fontsize=FSIZE_B)
    ax[1].set_title("Backpropagation Error",fontsize=FSIZE_B)

    # -- legend --
    # 0.55,0.45
    # .48,.195
    # leg0 = ax[0].legend(bbox_to_anchor=(0.0,0.90), loc="upper left",
    #                     fontsize=FSIZE_S,title="Exact",
    #                     title_fontsize=FSIZE_S,framealpha=1.,
    #                     edgecolor='k',ncol=2)
    # leg0.get_frame().set_alpha(None)
    # leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    solid_line = mlines.Line2D([], [], color='k', linestyle="-.",
                               marker="",label='In-Order')
    dotted_line = mlines.Line2D([], [], color='k', linestyle="-",
                                marker="",label='Random')
    h = [solid_line,dotted_line]
    leg2 = ax[1].legend(bbox_to_anchor=(-0.00,0.28), loc="upper left",
                        fontsize=FSIZE_S,title="Access Order",
                        title_fontsize=FSIZE_S,framealpha=1.,
                        edgecolor='k',ncol=1,handles=h)
    leg2.get_frame().set_alpha(None)
    leg2.get_frame().set_facecolor((0, 0, 0, 0.0))
    ax[1].add_artist(leg2)

    # print(handles)
    # print(h_labels)
    leg1 = ax[0].legend(bbox_to_anchor=(0.50,0.75),
                        loc="upper left",fontsize=FSIZE_S,
                        title="Channel Threads",framealpha=1.,
                        title_fontsize=FSIZE_S,edgecolor='k',ncol=2)
    leg1.get_frame().set_alpha(None)
    leg1.get_frame().set_facecolor((0, 0, 0, 0.0))


    # -- add text --
    ax[0].annotate("Exact GPU",xy=[5.,1.2],ha="center",fontsize=FSIZE_S)
    ax[0].annotate("Exact CPU",xy=[5.,.1],ha="center",fontsize=FSIZE_S)
    # ax[1].annotate("C Threads",xy=[15.,0.43],ha="center",fontsize=FSIZE_S)
    # ax[1].annotate("1 Thread",xy=[2.1,0.1],ha="left",fontsize=FSIZE_S)
    # ax[1].annotate("3 Threads",xy=[2.1,0.25],ha="left",fontsize=FSIZE_S)
    # ax[1].annotate("5 Threads",xy=[20.,0.2],ha="left",fontsize=FSIZE_S)

    # -- save figure --
    root = Path(str(SAVE_DIR) + "_plots")
    if not root.exists(): root.mkdir(parents=True)
    fn = root / "error_and_time.png"
    plt.savefig(str(fn),dpi=800,transparent=True)

def time_vs_channels(records):

    # -- only nbwd == 1 --
    # records = records[records["nbwd"].isin([1,15])]
    records = records.reset_index(drop=True)

    # -- make times log values --
    idx = records.index
    data = records.loc[:,'dtime'].astype(np.float32)
    # records.loc[idx,'dtime'] = np.log10(data)
    records.loc[idx,'dtime'] = np.log10(data)
    data = records.loc[:,'exact_time'].astype(np.float32)
    records.loc[idx,'exact_time'] = np.log10(data)

    # -- split data --
    edf = records[records['exact'] == True]
    # print(edf['dtime'])
    records = records[records['exact'] == False]
    # print(records[['dtime','exact_time','nbwd','nchnls','rbwd']])

    # -- order data --
    r0 = records[records['ngroups'] == 0]
    rg0 = records[records['ngroups'] > 0]
    records = rg0.append(r0)

    #
    # -- two side-by-side plots --
    #

    # -- plot constants --
    FSIZE = 12
    FSIZE_B = 14
    FSIZE_S = 10

    # -- init plot --
    ginfo = {'width_ratios': [0.47,.47],'wspace':0.05, 'hspace':0.0,
             "top":0.86,"bottom":0.16,"left":0.08,"right":0.99}
    # ginfo = {'width_ratios': [0.47,.47],'wspace':0.05, 'hspace':0.0,
    #          "top":0.80,"bottom":0.16,"left":0.115,"right":0.99}
    fig,ax = plt.subplots(1,2,figsize=(8,4),gridspec_kw=ginfo)

    # -- colors to nbwd --
    cmap = LinearSegmentedColormap.from_list("",["blue","yellow"])
    nunique_nbwd = len(records['ngroups'].unique())
    ax[0].set_prop_cycle(marker=['+', 'x', '*', '.', '^','s'])
    ax[1].set_prop_cycle(marker=['+', 'x', '*', '.', '^','s'])

    # -- create data to plot --
    i = 0
    for rbwd,rdf in records.groupby("rbwd"):
        b = 0
        for nbwd,bdf in rdf.groupby("ngroups",sort=False):
            # print(nbwd)

            # -- unpack --
            yvals0 = bdf['dtime']
            yvals1 = bdf['exact_time']
            xvals = bdf['nchnls']

            # -- plot --
            bwd_s = nbwd if i == 1 else None
            if bwd_s == 0: bwd_s = "C"
            col = b/float(nunique_nbwd)
            color = cmap(col)
            ax[i].plot(xvals, yvals0, color=color, label=bwd_s,
                       linewidth=2,#linestyle=lines[0],
                       markersize=10)
            # if b==1:
            #     label = "Exact" if i == 0 else None
            #     ax[i].plot(xvals, yvals1, color='k',label=label,
            #                linewidth=2,#linestyle=lines[1],
            #                markersize=10)
            b+=1
        i+=1

    # -- plot exact time --
    for i in range(len(ax)):
        xvals = edf['nchnls'].to_numpy()
        yvals_gpu = edf['dtime'].to_numpy()
        yvals_cpu = edf['exact_time'].to_numpy()
        order = np.argsort(xvals)
        xvals = xvals[order]
        yvals_cpu = yvals_cpu[order]
        yvals_gpu = yvals_gpu[order]
        label = "%s" % "CPU" if i == 0 else ""
        ax[i].plot(xvals,yvals_cpu,color='k',label=label,
                   markersize=10,linewidth=2)
        label = "%s" % "GPU" if i == 0 else ""
        ax[i].plot(xvals,yvals_gpu,color='k',label=label,
                   markersize=10,linewidth=2)

    # -- reset ticks --
    y_ours = records['dtime']
    y_exact = edf['dtime'] # exact cpu time
    x = records['nchnls'].to_numpy()
    x = np.sort(np.unique(x))
    xmin,xmax = x.min().item(),x.max().item()
    ymin,ymax = (y_ours*1.1).min().item(),(y_exact*1.1).max().item()
    yticks = np.linspace(ymin,ymax,5)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = x#np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]
    for i in range(len(ax)):
        if i == 0:
            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels(yticklabels,fontsize=FSIZE)
        else:
            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels([])
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticklabels,fontsize=FSIZE)
        ax[i].set_ylim(ymin,ymax)
    ax[0].set_ylabel("Log10 Wall-Clock Time (sec)",fontsize=FSIZE)
    ax[0].set_xlabel("Number of Channels (C)",fontsize=FSIZE)
    ax[1].set_xlabel("Number of Channels (C)",fontsize=FSIZE)

    # -- format titles --
    fig.suptitle("Comparing Execution Times",fontsize=FSIZE_B)
    # fig.suptitle("Fast, Approximate Backpropagation",fontsize=FSIZE_B)
    fig.suptitle("Reducing Race Condition Errors",fontsize=FSIZE_B)
    ax[0].set_title("Accessing Channels In-Order",fontsize=FSIZE_B)
    ax[1].set_title("Accessing Channels Randomly",fontsize=FSIZE_B)

    # -- legend --
    # 0.55,0.45
    leg0 = ax[0].legend(bbox_to_anchor=(0.48,0.195), loc="upper left",
                        fontsize=FSIZE_S,title="Exact",
                        title_fontsize=FSIZE_S,framealpha=1.,
                        edgecolor='k',ncol=2)
    leg0.get_frame().set_alpha(None)
    leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    leg1 = ax[1].legend(bbox_to_anchor=(0.59,0.27), loc="upper left",
                        fontsize=FSIZE_S,title="Channel Threads",
                        title_fontsize=FSIZE_S,framealpha=1.,
                        edgecolor='k',ncol=2)
    leg1.get_frame().set_alpha(None)
    leg1.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = Path(str(SAVE_DIR) + "_plots")
    if not root.exists(): root.mkdir(parents=True)
    fn = root / "time_vs_channels.png"
    plt.savefig(str(fn),dpi=800,transparent=True)


def errors_vs_channels(records):

    # -- clean-up --
    fields = ['errors_m','errors_s','nchnls','rbwd','nbwd']
    fields2drop = ['nchnls','rbwd','ngroups']
    records = records.drop_duplicates(fields2drop)
    records = records[records['exact'] == False]
    # print(records[fields])

    # -- only nbwd == 1 --
    # records = filter_records(records,{'nbwd':1})
    # records = records[records["nbwd"].isin([1,15])]
    # print(records[fields])

    #
    # -- two side-by-side plots --
    #

    # -- plot constants --
    FSIZE = 12
    FSIZE_B = 14
    FSIZE_S = 10

    # -- init plot --
    ginfo = {'width_ratios': [0.47,.47],'wspace':0.05, 'hspace':0.0,
             "top":0.86,"bottom":0.16,"left":0.08,"right":0.99}
    fig,ax = plt.subplots(1,2,figsize=(8,4),gridspec_kw=ginfo)

    # -- colors to nbwd --
    cmap = LinearSegmentedColormap.from_list("",["blue","yellow"])
    nunique_nbwd = len(records['ngroups'].unique())
    ax[0].set_prop_cycle(marker=['+', 'x', '*', '.', '^','s'])
    ax[1].set_prop_cycle(marker=['+', 'x', '*', '.', '^','s'])

    # -- create data to plot --
    i = 0
    records = records.sort_values(by="rbwd")
    for rbwd,rdf in records.groupby("rbwd"):
        b = 0
        for nbwd,bdf in rdf.groupby("ngroups"):
            # -- unpack --
            yvals = bdf['errors_m']
            yerr = bdf['errors_s']/np.sqrt(10.) # 10 == nreps
            xvals = bdf['nchnls']
            # print(rbwd,nbwd)

            # -- plot --
            bwd_s = nbwd
            # print(rbwd,i)
            col = b/float(nunique_nbwd)
            if i == 0:
                linestyle = "--"
                ax[1].errorbar(xvals, yvals, yerr=yerr,color=cmap(col),
                               label=bwd_s,linestyle=linestyle,
                               linewidth=2)
            linestyle = "-"
            ax[i].errorbar(xvals, yvals, yerr=yerr,color=cmap(col),
                           label=bwd_s,linestyle=linestyle,
                           linewidth=2)
            b+=1
        i+=1


    # -- reset ticks --
    y = records['errors_m']
    ybar = records['errors_s']
    x = records['nchnls'].to_numpy()
    x = np.sort(np.unique(x))
    xmin,xmax = x.min().item(),x.max().item()
    ymin,ymax = (y-ybar/1.2).min().item(),(y+ybar/2.).max().item()
    yticks = np.linspace(ymin,ymax,5)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = x#np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]
    for i in range(2):
        if i == 0:
            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels(yticklabels,fontsize=FSIZE)
        else:
            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels([])
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticklabels,fontsize=FSIZE)
        ax[i].set_ylim(ymin,ymax)
    ax[0].set_ylabel("Relative Error",fontsize=FSIZE)
    ax[0].set_xlabel("Number of Channels",fontsize=FSIZE)
    ax[1].set_xlabel("Number of Channels",fontsize=FSIZE)

    # -- format titles --
    fig.suptitle("Reducing Race Condition Errors",fontsize=FSIZE_B)
    ax[0].set_title("Accessing Channels In-Order",fontsize=FSIZE_B)
    ax[1].set_title("Accessing Channels Randomly",fontsize=FSIZE_B)

    # -- legend --
    leg1 = ax[1].legend(bbox_to_anchor=(0.65,1.01), loc="upper left",fontsize=FSIZE,
                        title="Num. Bkwd",title_fontsize=FSIZE,framealpha=1.,
                        edgecolor='k')
    leg1.get_frame().set_alpha(None)
    leg1.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = Path(str(SAVE_DIR) + "_plots")
    if not root.exists(): root.mkdir(parents=True)
    fn = root / "errors_vs_channels.png"
    # print("Save: ",fn)
    plt.savefig(str(fn),dpi=800,transparent=True)
