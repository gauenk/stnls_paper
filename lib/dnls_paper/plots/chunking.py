"""

Showing the impact of cropping

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

# -- interpolate --
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
from scipy.interpolate import interpn,RegularGridInterpolator,RBFInterpolator
from scipy.interpolate import NearestNDInterpolator

# -- plotting --
import matplotlib.pyplot as plt

# -- const --
SAVE_DIR = Path("./output/plots/chunking/")
FSIZE = 12
FSIZE_B = 14
FSIZE_T = 12
FSIZE_S = 12


def run(records,arch):
    fields = ["psnrs","strred","timer_deno","deno_mem_res"]
    names = ["PSNR","ST-RRED","Runtime","Memory"]
    units = ["dB","$10^{-2}$","seconds","GB"]
    chunk_fields = []
    for fname in ["spatial","temporal"]:
        for ftype in ["size","overlap"]:
            chunk_fields.append("%s_chunk_%s" % (fname,ftype))
    all_fields = fields + chunk_fields + ["attn_timer"]
    records = records[all_fields]

    if arch == "lidia":
        df = records[~np.isnan(records['spatial_chunk_size'])]
        df.drop_duplicates(inplace=True)
    else:
        df = records[records['attn_timer'] == False]

    plot_chunks_vs_overlap_vs_field(df,"spatial","Resolution Chunk Size",
                                    fields,names,units,arch)
    # df = records[records['attn_timer'] == True]

    if arch == "lidia":
        df = records[records['temporal_chunk_size']>0]
        df.drop_duplicates(inplace=True)
    else:
        df = records[records['attn_timer'] == True]

    plot_chunks_vs_overlap_vs_field(df,"temporal","Frame Chunk Size",
                                    fields,names,units,arch)
    # plot_schunks_vs_overalp_vs_field(records,"spatial",fields,names,units)

def plot_chunks_vs_overlap_vs_field(records,chunk_field,xlabel,fields,names,units,arch):

    # -- unpack --
    chunk = records['%s_chunk_size' % chunk_field].to_numpy()
    olap = records['%s_chunk_overlap' % chunk_field].to_numpy()
    NF = len(fields)

    # -- create figure --
    wr = 1./NF-0.01
    ginfo = {'width_ratios': [wr,]*4,'wspace':0.12, 'hspace':0.0,
             "top":0.80,"bottom":0.23,"left":0.07,"right":0.98}
    fig,axes = plt.subplots(1,NF,figsize=(2.5*NF,2),gridspec_kw=ginfo)

    # -- create mesh --
    X = np.linspace(min(chunk), max(chunk))
    Y = np.linspace(min(olap), max(olap))
    mX, mY = np.meshgrid(X, Y)  # 2D grid for interpolation

    # -- plot each field --
    for i,field in enumerate(fields):

        # -- mesh field --
        Z = records[field].to_numpy()
        interp = LinearNDInterpolator(list(zip(chunk, olap)), Z)
        mZ = interp(mX,mY)

        # -- plot --
        im = axes[i].pcolormesh(mX, mY, mZ, shading="auto",label=names[i])

        # -- colorbar config --
        bname = names[i] + " (%s)"%units[i]
        if field == "psnr":
            bname += r" $\uparrow$"
        elif field == "strred":
            bname += r" $\downarrow$"
        cbar = fig.colorbar(im,ax=axes[i],orientation="horizontal",
                            location="top",extend='both')
        cbar.set_label(bname,size=FSIZE)
        ticklabs = cbar.ax.get_xticklabels()
        amin,amax = mZ.min(),mZ.max()
        ticks = np.linspace(amin,amax,num=4,endpoint=True)
        if field == "strred":
            ticklabs = ["%2.1f" % (x*100) for x in ticks]
        else:
            ticklabs = ["%2.1f" % x for x in ticks]
        cbar.ax.set_xticks(ticks)
        cbar.ax.set_xticklabels(ticklabs, fontsize=FSIZE)

    # -- clear tickmarks --
    ylabel = "Overlap"
    plot_spatial_set_ticks(axes,chunk,olap,xlabel,ylabel)

    # -- titles --

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = root / ("%s_%s.png" % (arch,chunk_field))
    plt.savefig(str(fn),dpi=300,transparent=True)
    plt.close("all")

def plot_spatial_set_ticks(axes,x,y,xlabel,ylabel):

    # -- get ticks --
    xmin,xmax = x.min().item(),x.max().item()
    ymin,ymax = y.min().item(),y.max().item()
    yticks = np.unique(y)#np.linspace(ymin,ymax,5)
    yticklabels = ["%d%%" % (x*100) for x in yticks]
    xticks = np.unique(x)#np.unique(x.copy().tolist())#np.linspace(xmin,xmax,4)
    xticks = list(xticks)
    xticklabels = ["%d" % x for x in xticks]

    # -- reset axis --
    for i,ax in enumerate(axes):
        if i == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels,fontsize=FSIZE)
            ax.set_ylabel(ylabel,fontsize=FSIZE)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
        ax.set_xlabel(xlabel,fontsize=FSIZE)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels,fontsize=FSIZE)
        ax.set_ylim(ymin,ymax)
