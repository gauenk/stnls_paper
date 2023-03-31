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
SAVE_DIR = Path("./output/plots/modulate_gpu_runtime/")


def run(records):

    # -- create search total --
    ws = records['ws']
    wt = records['wt']
    records['nsearch'] = ws*ws * (2*wt + 1)
    records['nsearch_t'] = (2*wt + 1)
    records = records.rename(columns={"deno_mem_res":"mem_res",
                                      "bs":"batch_size"})

    # -- filter (ws,wt) --
    # records = records[records['ws'] == 20]
    # records = records[records['wt'] == 3]

    # -- plot constants --
    FSIZE = 18
    FSIZE_B = 18
    FSIZE_S = 15

    # -- init plot --
    ginfo = {'width_ratios': [1.],'wspace':0.05, 'hspace':0.0,
             "top":0.88,"bottom":0.23,"left":0.115,"right":0.97}
    fig,ax = plt.subplots(1,1,figsize=(8,2.8),gridspec_kw=ginfo)

    # -- colors to nbwd --
    lines = ['-','--']
    colors = {"lidia":"red","colanet":"orange","n3net":"blue"}
    markers = ["^","x","*"]
    order = ["lidia","colanet","n3net"]
    labels = {"lidia":"LIDIA","colanet":"COLA-Net","n3net":"N3Net"}

    # -- unpack info --
    ix = 0
    for mname in order:
        # for mname,df in records.groupby("arch_name"):
        df = records[records['arch_name'] == mname]
        name = labels[mname]#df['label_name'].iloc[0]
        sb = np_col(df['batch_size']).astype(np.int)
        order = np.argsort(sb)
        sb = sb[order]
        xvals = np_col(df['timer_deno'])[order]
        yvals = np_col(df['mem_res'])[order]
        # print(sb,xvals,yvals)
        ax.plot(xvals,yvals,linewidth=2,linestyle='-',
                markersize=10,marker=markers[ix],label=name,
                color=colors[mname])
        ix += 1

    # -- reset ticks --
    y = np.stack(records['mem_res'].to_numpy()).ravel()
    x = np.stack(records['timer_deno'].to_numpy()).ravel()
    # print(y,x)
    # print(y.shape,x.shape)
    xmin,xmax = 0.9*x.min().item(),1.1*x.max().item()
    ymin,ymax = 0.1*y.min().item(),1.03*y.max().item()
    yticks = np.linspace(ymin,ymax,5)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]
    # for i in range(2):
    #     if i == 0:
    #         ax[i].set_yticks(yticks)
    #         ax[i].set_yticklabels(yticklabels,fontsize=FSIZE)
    #     else:
    #         ax[i].set_yticks(yticks)
    #         ax[i].set_yticklabels([])
    #     ax[i].set_xticks(xticks)
    #     ax[i].set_xticklabels(xticklabels,fontsize=FSIZE)
    #     ax[i].set_ylim(ymin,ymax)

    # -- set ticks --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    ax.set_ylim(ymin,ymax)

    # -- set labels --
    ax.set_ylabel("GPU Memory (GB)",fontsize=FSIZE)
    ax.set_xlabel("Wall-Clock Time (sec)",fontsize=FSIZE)
    # ax[1].set_xlabel("Num. of Channels",fontsize=FSIZE)

    # -- format titles --
    # fig.suptitle("Comparing Execution Times",fontsize=FSIZE_B)
    ax.set_title("Modulating Memory and Wall-Clock Time",fontsize=FSIZE_B)
    # ax[1].set_title("Randomized Channels",fontsize=FSIZE_B)

    # -- legend --
    leg0 = ax.legend(bbox_to_anchor=(0.668,1.05), loc="upper left",fontsize=FSIZE,
                     title="Network",title_fontsize=FSIZE,framealpha=1.,
                     edgecolor='k')
    leg0.get_frame().set_alpha(None)
    leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "mem_vs_runtime.png")
    print("Saving plot at %s" % fn)
    plt.savefig(str(fn),dpi=800,transparent=True)

def np_col(df_label):
    return np.stack(df_label).ravel()

