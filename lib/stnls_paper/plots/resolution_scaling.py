"""

Trade-off GPU Memory and Runtime via batch size

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- management --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
pd.options.mode.chained_assignment = None  # default='warn'

# -- plotting --
import matplotlib.pyplot as plt

# -- const --
SAVE_DIR = Path("./output/plots/resolution_scaling/")

def run(records):
    records = prepare_records(records)
    # for arch_name,adf in records.groupby("arch_name"):
    triplet(records)
    # res_vs_field(records,'dtime','Runtime (seconds)')
    # res_vs_field(records,'stime','Runtime (seconds)')
    # res_vs_field(records,'mem','Memory (GB)')

def triplet(records):

    # -- init --
    # ginfo = {'wspace':0.5, 'hspace':0.2,
    #          "top":0.98,"bottom":0.07,"left":0.20,"right":0.99}
    # fig,axes = plt.subplots(3,1,figsize=(5,8),gridspec_kw=ginfo)

    ginfo = {'wspace':0.38, 'hspace':0.2,
             "top":0.93,"bottom":0.17,"left":0.08,"right":0.99}
    fig,axes = plt.subplots(1,3,figsize=(12,4),gridspec_kw=ginfo)

    # -- get ylims --
    ylims = get_ylims(records)

    # -- plot --
    res_vs_field(records,'dtime','Denoise Time (sec)',axes[0],2,True)
    res_vs_field(records,'ptime','Percent of Time Searching',axes[1],2,False)
    # res_vs_field(records,'stime','Search Time (sec)',axes[1],2,False,ylims=ylims)
    res_vs_field(records,'mem','Memory (GB)',axes[2],2,False)

    # -- save --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / ("scaling_res_triplet.png"))
    print("Saving figure %s" % fn)
    plt.savefig(fn,dpi=500,transparent=True)
    plt.close("all")

def res_vs_field(records,field,ylabel,in_axis=None,axis_id=-1,plt_lgnd=True,ylims=None):

    # -- plot constants --
    FSIZE = 18
    FSIZE_B = 18
    FSIZE_S = 15

    # -- no axis --
    if in_axis is None:
        ginfo = {'width_ratios': [1.],'wspace':0.1, 'hspace':0.0,
                 "top":0.88,"bottom":0.2,"left":0.14,"right":0.99}
        fig,ax = plt.subplots(1,1,figsize=(8,2.8),gridspec_kw=ginfo)
    else:
        ax = in_axis
        axis_id = -1 if (axis_id == 2) else axis_id

    # -- colors to nbwd --
    lines = ['-','--']
    colors = {"lidia":"red","colanet":"orange","n3net":"blue"}
    markers = ["^","x","*"]
    order = ["colanet","n3net","lidia"]
    order_2 = [2, 0, 1]
    labels = {"lidia":"LIDIA","colanet":"COLA-Net","n3net":"N3Net"}

    # -- unpack info --
    ix = 0
    for mname in order:
        # for mname,df in records.groupby("arch_name"):
        df = records[records['arch_name'] == mname]
        name = labels[mname]#df['label_name'].iloc[0]
        res = df['res'].to_numpy()
        aorder = np.argsort(res)
        res = res[aorder]
        xvals = res
        yvals = df[field].to_numpy()[aorder]
        ax.plot(xvals,yvals,linewidth=2,linestyle='-',
                markersize=10,marker=markers[ix],label=name,
                color=colors[mname])
        ix += 1

    # -- reset ticks --
    y = records[field].to_numpy()
    x = np.unique(records['res'].to_numpy())
    xmin,xmax = 0.9*x.min().item(),1.1*x.max().item()
    xticks = np.r_[x[1],x[-2],x[-1]]#np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]

    if "dtime" in field:
        # data = np.log10(data)
        # ymin,ymax = data.min(),data.max()
        # ylims[gname] = [10**(ymin*1.)*.9,10**(ymax*1.)*1.1] # pos,pos
        # yticks[gname] = [v for v in np.logspace(ymin,ymax,5)]
        # yticklabels[gname] = ["%2.1e" % s for s in yticks[gname]]
        # yticks_m[gname] = np.logspace(ymin,ymax,25)
        yl = np.log10(y)
        ymin,ymax = yl.min().item(),yl.max().item()
        yticks = np.logspace(ymin,ymax,5)
        yticklabels = ["%1.1f" % x for x in yticks]
        yticks_m = np.logspace(ymin,ymax,25)
        ymin,ymax = .9*y.min().item(),1.1*y.max().item()
        ax.set_yscale("log")
    elif "ptime" in field:
        yticks = np.linspace(0,100,5)
        yticklabels = ["%2.0f" % x for x in yticks]
        yticks_m = []
    else:
        ymin,ymax = 0.1*y.min().item(),1.03*y.max().item()
        yticks = np.linspace(ymin,ymax,5)
        yticklabels = ["%1.1f" % x for x in yticks]
        yticks_m = []

    # -- set ticks --
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    if "dtime" in field:
        blank = ["" for _ in range(len(yticks_m))]
        ax.set_yticks(yticks_m,labels=blank,minor=True)
    if "ptime" in field:
        ax.set_ylim(0,100)
    elif ylims is None:
        ax.set_ylim(ymin,ymax)
    else:
        ax.set_ylim(ylims[0],ylims[1])
    ax.set_ylabel(ylabel,fontsize=FSIZE)

    # -- set labels --
    if axis_id == -1:
        ax.set_xlabel("Image Resolution (pixels)",fontsize=FSIZE)

    # -- legend --
    if in_axis is None or plt_lgnd:
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[o] for o in order_2]
        labels = [labels[o] for o in order_2]
        # loc = (0.51,.75)
        loc = (0.35,.48)
        leg0 = ax.legend(bbox_to_anchor=loc, loc="upper left",fontsize=FSIZE_S,
                         title="Network",title_fontsize=FSIZE,framealpha=1.,
                         edgecolor='k',labels=labels,handles=handles)
        leg0.get_frame().set_alpha(None)
        leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    if in_axis is None:
        root = SAVE_DIR
        if not root.exists(): root.mkdir(parents=True)
        fn = root / ("scaling_res_%s.png" % field)
        plt.savefig(str(fn),dpi=800,transparent=True)
        plt.close("all")


def prepare_records(records):
    ws = records['ws']
    wt = records['wt']
    records['nsearch'] = ws*ws * (2*wt + 1)
    records['nsearch_t'] = (2*wt + 1)
    records = records.rename(columns={"deno_mem_res":"mem",
                                      "timer_deno":"dtime",
                                      "timer_search":"stime"})
    # -- select --
    fields = ["res","dtime","stime","mem",'arch_name']
    records['res'] = records['isize'].str.split("_",expand=True)[0]
    records['res'] = records['res'].astype(int)
    records['dtime'] = records['dtime'].apply(lambda x:x[0])
    records['stime'] = records['stime'].apply(lambda x:np.array(x[0])).apply(np.sum)
    records['ptime'] = records['stime']/records['dtime']*100
    records['mem'] = records['mem'].apply(lambda x:x[0][0])
    return records

def get_ylims(records):
    dtime = records['dtime'].to_numpy()
    stime = records['stime'].to_numpy()
    times = np.r_[dtime,stime]
    ymin = times.min()
    ymax = times.max()
    return ymin,ymax

def np_col(df_label):
    return np.stack(df_label).ravel()

