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

# -- interpolate --
from scipy.interpolate import LinearNDInterpolator

# -- const --
SAVE_DIR = Path("./output/plots/search_space/")

def run(records,field,field_label):

    # -- create plotting fields --
    arch_name = str(records['arch_name'].iloc[0])
    records = prepare_records(records)
    records = records[['ws','st',field]]
    # print(records.head())
    # print(records.columns)

    # -- plot constants --
    FSIZE = 18
    FSIZE_B = 18
    FSIZE_S = 15

    # -- init plot --
    ginfo = {'width_ratios': [1.],'wspace':0.05, 'hspace':0.0,
             "top":0.86,"bottom":0.18,"left":0.10,"right":0.97}
    fig,ax = plt.subplots(1,1,figsize=(7,4),gridspec_kw=ginfo)

    # -- plot --
    ws = records['ws'].to_numpy()
    st = records['st'].to_numpy()
    Z = records[field].to_numpy()
    X = np.linspace(min(ws), max(ws))
    Y = np.linspace(min(st), max(st))
    mX, mY = np.meshgrid(X, Y)  # 2D grid for interpolation
    # print(ws,st,X,Y,Z)
    interp = LinearNDInterpolator(list(zip(ws,st)), Z)
    mZ = interp(mX,mY)

    # -- plot --
    im = ax.pcolormesh(mX, mY, mZ, shading="auto")#,label=field_label)

    # -- colorbar config --
    bname = field_label
    cbar = fig.colorbar(im,ax=ax,orientation="horizontal",
                        location="top",extend='both')
    cbar.set_label(bname,size=FSIZE)
    ticklabs = cbar.ax.get_xticklabels()
    amin,amax = mZ.min(),mZ.max()
    ticks = np.linspace(amin,amax,num=4,endpoint=True)
    if field == "strred":
        ticklabs = ["%2.1f" % x for x in ticks]
    else:
        ticklabs = ["%2.1f" % x for x in ticks]
    cbar.ax.set_xticks(ticks)
    cbar.ax.set_xticklabels(ticklabs, fontsize=FSIZE)


    # -- reset ticks --
    x = ws
    y = st
    xmin,xmax = 1.*x.min().item(),1.*x.max().item()
    ymin,ymax = 1.*y.min().item(),1.*y.max().item()
    yticks = np.linspace(ymin,ymax,4)
    yticklabels = ["%d" % x for x in yticks]
    xticks = np.linspace(xmin,xmax,5)
    xticklabels = ["%d" % x for x in xticks]

    # -- set ticks --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    ax.set_ylim(ymin,ymax)

    # -- set labels --
    ax.set_ylabel("Time Window (frames)",fontsize=FSIZE)
    ax.set_xlabel("Space Window (pixels)",fontsize=FSIZE)

    # -- format titles --
    # ax.set_title("Runtime v.s. Resolution",fontsize=FSIZE_B)

    # -- legend --
    # handles, labels = ax.get_legend_handles_labels()
    # handles = [handles[o] for o in order_2]
    # labels = [labels[o] for o in order_2]
    # leg0 = ax.legend(bbox_to_anchor=(0.668,1.05), loc="upper left",fontsize=FSIZE,
    #                  title="Network",title_fontsize=FSIZE,framealpha=1.,
    #                  edgecolor='k',labels=labels,handles=handles)
    # leg0.get_frame().set_alpha(None)
    # leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / ("%s_%s.png" % (arch_name,field)))
    print("Saving plot at %s" % fn)
    plt.savefig(fn,dpi=300,transparent=True)

def prepare_records(records):
    # -- create search total --
    ws = records['ws']
    wt = records['wt']
    records['nsearch'] = ws*ws * (2*wt + 1)
    records['nsearch_t'] = (2*wt + 1)
    records['st'] = (2*wt + 1)
    records = records.rename(columns={"deno_mem_res":"mem",
                                      "timer_deno":"time",
                                      "timer_search":"stime"})

    # -- group across video frames --
    fields = ['ws','wt','st','arch_name','isize']
    res_fields = ['time','mem','psnrs','strred']
    agg = {key:[np.stack] for key in res_fields}
    grouped = records.groupby(fields).agg(agg)
    grouped.columns = res_fields
    grouped = grouped.reset_index()
    for field in res_fields:
        grouped[field] = grouped[field].apply(np.mean)
    records = grouped
    # print(records.head())
    records['strred'] = records['strred']*100

    # -- select --
    # records['res'] = records['isize'].str.split("_",expand=True)[0]
    # records['res'] = records['res'].astype(int)
    # records['time'] = records['dtime'].apply(lambda x:x[0])
    # records['stime'] = records['stime'].apply(np.sum)
    # records['mem'] = records['mem_res'].apply(lambda x:x[0][0])
    # records['psnrs'] = records['psnrs'].apply(np.ravel).apply(np.mean)
    # records['dtime'] = records['dtime'].apply(np.mean)
    # records['strred'] = records['strred'].apply(np.ravel).apply(np.mean)*100.
    return records

def np_col(df_label):
    return np.stack(df_label).ravel()

