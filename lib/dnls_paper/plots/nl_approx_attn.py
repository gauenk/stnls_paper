# -- misc --
import os,math,tqdm
import pprint,random
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- matplotlib --
import matplotlib.pyplot as plt
SAVE_DIR = Path("output/plots/nls_approx_attn/")


# -- plot constants --
FSIZE = 15
FSIZE_B = 16
FSIZE_S = 13

def run(records):
    # -- init plot --
    ginfo = {'wspace':0.2, 'hspace':0.0,
             "top":0.88,"bottom":0.14,"left":0.115,"right":0.99}
    fig,axes = plt.subplots(1,2,figsize=(9,4),gridspec_kw=ginfo)

    # -- plot --
    records['error_m'] = np.log10(records['error_m'])
    error_vs_neigh(records,axes[0],0)
    error_vs_neigh(records,axes[1],1)

    # -- sup titles --
    # fig.suptitle("Non-Local Search Approximates Global Attention",fontsize=FSIZE_B)

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "nls_approx_attn.png")
    print("Saving figure %s" % fn)
    plt.savefig(fn,dpi=500,transparent=True)

def error_vs_neigh(records,ax,ax_id):
    # -- log --
    # records = records.reset_index(drop=True)
    # idx = records.index
    records = records.sort_values(by=["ws","k_s"],ascending=True)
    # records['error_m'] = np.log10(records['error_m'])
    # data = records.loc[:,'errors_m'].astype(np.float32)
    # records.loc[idx,'errors_m'] = np.log10(data)
    # data = records.loc[:,'errors_s'].astype(np.float32)
    # records.loc[idx,'errors_s'] = np.log10(data)

    # -- colors to nbwd --
    lines = ['-','--']
    colors = ["blue","orange","purple"]
    Z = np.sqrt(3*128*128)

    # -- two types --
    b = 0
    ws_order = [-1,27,21]
    for ws in ws_order:
    # for ws,cdf in records.groupby("ws"):

        # -- unpack --
        cdf = records[records['ws'] == ws]
        yvals = np.stack(cdf['error_m'].to_numpy()).ravel()
        # yerr = np.stack(cdf['errors_s'].to_numpy()).ravel()/Z
        xvals = np.stack(cdf['k_a'].to_numpy()).ravel()

        # -- plot --
        # label = "Attn." if run_ca_fwd == "true" else "Final"
        label = "Global" if ws == -1 else ws
        color = colors[b]
        # ax.errorbar(xvals, yvals, yerr=yerr,color=color, label=label,
        #             linewidth=3)
        ax.plot(xvals, yvals, color=color, label=label,linewidth=3,
                marker='x',markersize=10)
        b+=1

    # -- compute ticks --
    if ax_id == 0:
        y = records['error_m']
        x = records['k_a'].to_numpy()
    else:
        df = records[records['ws'].isin([21,27])]
        y = df['error_m']
        x = df['k_a'].to_numpy()
    x = np.sort(np.unique(x))
    xmin,xmax = 0,x.max().item()
    ymin,ymax = y.min().item(),y.max().item()
    yticks = np.linspace(ymin,ymax,5)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]

    # -- set ticks --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    if ax_id == 0:
        ax.set_ylim(ymin*1.1,ymax*1.3)
    else:
        ax.set_ylim(ymin*.4,ymax*1.05)
    if ax_id == 0:
        ax.set_xlim(-1000,x.max()*1.1)
    else:
        ax.set_xlim(-50,x.max()*1.1)

    # -- set labels --
    if ax_id == 0:
        ax.set_ylabel("Log10 Relative Error",fontsize=FSIZE)
    ax.set_xlabel("Number of Neighbors",fontsize=FSIZE)
    # ax.axhline(np.log10(1e-5),color='k',linestyle='--')

    if ax_id == 0:
        ax.set_title("A Global Search Approximates Attention",fontsize=FSIZE_B)
    elif ax_id == 1:
        ax.set_title("A Local Search Yields Differences",fontsize=FSIZE_B)

    # -- legend --
    # 0.65,1.08
    if ax_id == 1:
        leg1 = ax.legend(bbox_to_anchor=(0.3,0.99), loc="upper left",fontsize=FSIZE_S,
                         title="Spatial Window",title_fontsize=FSIZE_S,framealpha=1.,
                         edgecolor='k',ncol=2)
        leg1.get_frame().set_alpha(None)
        leg1.get_frame().set_facecolor((0, 0, 0, 0.0))
