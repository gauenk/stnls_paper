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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# -- const --
SAVE_DIR = Path("./output/plots/race_cond/")
FSIZE = 12
FSIZE_B = 14
FSIZE_T = 12
FSIZE_S = 12

def run(records):
    # print(records[['neigh_pt','query_pt']].drop_duplicates())
    plot_heatmap_sets(records)


def plot_heatmap_sets(records):

    # -- prepare --
    records = prepare_records(records)

    # -- settings --
    axis_label = True
    tgts = ["error","time"]
    tgt_labels = ["Relative Error","Time"]
    tgt_units = ["","seconds"]

    fields = ['neigh_pt','query_pt','chnls_pt','nchnls']
    fields_labels = ["Neighbors Per Thread","Queries Per Thread",
                     "Channels Per Thread","Number of Channels"]

    fixed = {"neigh_pt":4,"query_pt":4,"chnls_pt":1,"nchnls":30}


    # -- heatmaps --
    # df_ = records[records['rbwd'] == True]
    # plot_heatmaps(df_,"rand",cbar_lims,fixed,
    #               tgts,tgt_labels,tgt_units,
    #               fields,fields_labels,axis_label)
    # df_ = records[records['rbwd'] == False]

    # -- init --
    height_ratios = [1.]*6 + [0.1]
    ginfo = {'wspace':0.2, 'hspace':0.4,"height_ratios":height_ratios,
             "top":.98,"bottom":0.04,"left":0.05,"right":0.99}
    # fig,axes = plt.subplots(6,4,figsize=(9,14),gridspec_kw=ginfo)
    fig = plt.figure(figsize=(9,14))
    axes,cbars_ax = create_gridspec(fig,ginfo)
    set_titles(axes)

    # -- create colorbar --
    cbar_lims = get_color_bar_lims(records,tgts,fixed,fields)
    plot_colorbar(cbar_lims['error'],'error',cbars_ax[0])
    plot_colorbar(cbar_lims['time'],'time',cbars_ax[1])

    # -- plot --
    plot_heatmaps(records,"full",cbar_lims,fixed,
                  tgts,tgt_labels,tgt_units,
                  fields,fields_labels,axis_label,fig,axes)

    # -- save --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = root / "heatmap_grid.png"
    print("Saving figure %s" % str(fn))
    plt.savefig(str(fn),dpi=300,transparent=True)
    plt.close("all")

def set_titles(axes):
    for i in range(4):
        if i % 2 == 0:
            axes[0][i].set_title("In-Order",fontsize=FSIZE)
        else:
            axes[0][i].set_title("Randomly",fontsize=FSIZE)

def create_gridspec(fig,ginfo):

    # create sub plots as grid
    ginfo['figure'] = fig
    gs = GridSpec(7, 4, **ginfo)
    axes = []
    for i in range(6):
        axes_i = []
        for j in range(4):
            axes_i.append(fig.add_subplot(gs[i, j]))
        axes.append(axes_i)

    cbars = []
    cbars.append(fig.add_subplot(gs[6, :2]))
    cbars.append(fig.add_subplot(gs[6, -2:]))
    return  axes,cbars

def plot_heatmaps(records,postfix,cbar_lims,fixed,tgts,tgt_labels,
                  tgt_units,fields,fields_labels,axis_label,fig,axes):
    # fields = ['neigh_pt','query_pt','ngroups','nchnls']
    # fields = [fields[i] for i in range(2,4)]
    # fields_labels = [fields_labels[i] for i in range(2,4)]
    axis_label = True
    a0,a1 = 0,0
    for i0,(f0,f0_l) in enumerate(zip(fields,fields_labels)):
        for i1,(f1,f1_l) in enumerate(zip(fields,fields_labels)):
            if i0 >= i1: continue
            for tgt,tgt_l,tgt_u in zip(tgts,tgt_labels,tgt_units):
                for rbwd in [False,True]:
                    _df = filter_fixed(records,f0,f1,fixed,rbwd)
                    ax = axes[a0][a1]
                    if a1 == 0: axis_label = True
                    else: axis_label = False
                    # print(a0,a1,f0,f1,tgt)
                    p = get_params(f0,f1,axis_label)
                    _df = modded_for_fields(_df,f0,f1)
                    plot_pair_heatmap(_df,f0,f0_l,f1,f1_l,tgt,
                                      tgt_l,tgt_u,p,postfix,cbar_lims,
                                      axis_label,ax)
                    a1 += 1
                    if a1 % 4 == 0:
                        share_xlabel(fig,axes,a0,f0_l)
                        a0 += 1
                        a1 = 0

def modded_for_fields(_df,f0,f1):
    if f0 == "chnls_pt":
        chnls_pt = _df['ngroups'].to_numpy()
        chnls_pt = np.where(chnls_pt == 0,100,chnls_pt)
        _df['ngroups'] = chnls_pt
    return _df

def share_xlabel(fig,axes,a0,f0):
    if f0 == "Channels Per Thread":
        f0 = "Number of Channel Groups"
    label= "<"+"-"*10 + " " + f0 + " " + "-"*10 + ">"
    x = .335
    y = 0
    for ai in range(len(axes)):
        y = axes[ai][0].bbox._bbox.y0
        if ai == a0: break
    y -= 0.030
    fig.text(x,y,label,fontsize=FSIZE)

def get_params(f0,f1,axis_label):
    p = get_params_v0(f0,f1)
    if axis_label is True:
        return p
    p.left -= 0.1
    p.btm -= 0.08
    return p

def get_params_v0(f0,f1):
    params = edict()
    params.yshift = 0.5
    if f0 == "neigh_pt" and f1 in ["chnls_pt","nchnls"]:
        params.top = 0.97
        params.btm = .19
        params.left = .2
        params.right = .99
        params.fsize = (2.3,2.5)
    elif f0 == "neigh_pt" and f1 == "query_pt":
        params.top = 0.98
        params.btm = .20
        params.left = .15
        params.right = .99
        params.fsize = (2.5,2.5)
    elif f0 == "query_pt":
        params.top = 0.97
        params.btm = .2
        params.left = .2
        params.right = .99
        params.fsize = (2.2,2.5)
    elif f0 == "chnls_pt":
        params.top = 0.97
        params.btm = .2
        params.left = .17
        params.right = .99
        params.fsize = (2.7,2.5)
        # params.top = 0.98
        # params.btm = .20
        # params.left = .15
        # params.right = .95
        # params.fsize = (4.5,2)
        # params.yshift = 0.45
    else:
        params.top = 0.94
        params.btm = .20
        params.left = .01
        params.right = .88
        params.fsize = (3,2.5)
    return params

def plot_pair_heatmap(df,f0,f0_label,f1,f1_label,tgt,tgt_label,
                      tgt_units,p,postfix,cbar_lims,axis_label,in_ax):


    # -- fix chnls_pt -> ngroups if (nchnls,chnls_pt) --
    if f0 == "chnls_pt" and f1 == "nchnls":
        f0 = "ngroups"
        f0_label = "Number of Channel Groups"

    # -- create mesh --
    X = df[f0].to_numpy()
    Y = df[f1].to_numpy()
    Z = df[tgt].to_numpy()
    mX,mY,mZ = get_mesh(X,Y,Z)
    # print(mZ.min().item(),mZ.max().item())

    # -- init --
    if in_ax is None:
        ginfo = {'width_ratios': [1.],'wspace':0.0, 'hspace':0.0,
                 "top":p.top,"bottom":p.btm,"left":p.left,"right":p.right}
        fig,ax = plt.subplots(1,1,figsize=p.fsize,gridspec_kw=ginfo)
    else:
        ax = in_ax

    # -- plot --
    if tgt == "time":
        color = "PiYG"
        norm = mcolors.LogNorm(vmin=cbar_lims[tgt][0],
                               vmax=cbar_lims[tgt][1])
    else:
        color = "coolwarm"
        norm = mcolors.Normalize(vmin=cbar_lims[tgt][0],
                                 vmax=cbar_lims[tgt][1])

    im = ax.pcolormesh(mX, mY, mZ, norm=norm, cmap=color) # shading="auto",

    # -- format colorbar --
    # format_colorbar(fig,ax,im,mZ,tgt_label,tgt_units,cbar_lims[tgt])

    # -- clear tickmarks --
    xlabel = f0_label
    ylabel = f1_label
    # print(X.min(),X.max(),f0_label)
    set_ticks(ax,X,Y,xlabel,ylabel,p,axis_label)
    ax.set_aspect('equal')

    # -- save figure --
    if in_ax is None:
        root = SAVE_DIR
        if not root.exists(): root.mkdir(parents=True)
        fn = root / ("%s_%s_%s_%s.png" % (f0,f1,tgt,postfix))
        plt.savefig(str(fn),dpi=300,transparent=True)
        plt.close("all")

def format_colorbar(fig,ax,im,mZ,name,units):
    bname = name
    if units != "":
        bname += " (%s)" % units
    cbar = fig.colorbar(im,ax=ax,orientation="vertical",
                        extend='both',
                        norm=mcolors.LogNorm())
    cbar.set_label(bname,size=FSIZE)
    ticklabs = cbar.ax.get_yticklabels()
    amin,amax = cbar_lims[0],cbar_lims[1]
    # amin,amax = np.nanmin(mZ),np.nanmax(mZ)
    ticks = np.linspace(amin,amax,num=4,endpoint=True)
    ticklabs = ["%2.2f" % x for x in ticks]
    cbar.ax.set_yticks(ticks)
    cbar.ax.set_yticklabels(ticklabs, fontsize=FSIZE)


def set_ticks(ax,x,y,xlabel,ylabel,p,axis_label):
    set_yticks(ax,y,ylabel,p,axis_label)
    if xlabel == "Number of Channel Groups":
        set_xticks_v0(ax,x,xlabel)
    else:
        set_xticks_def(ax,x,xlabel)

def set_yticks(ax,y,ylabel,p,axis_label):
    ymin,ymax = y.min().item(),y.max().item()
    nuniq = len(np.unique(y))
    yticks = np.arange(nuniq)#np.linspace(ymin,ymax,5)
    yticks_l = np.linspace(ymin,ymax,nuniq)
    yticklabels = ["%d" % x for x in yticks_l]
    ymin,ymax = 0-0.5,nuniq-0.5

    # -- set axis --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    if axis_label:
        ax.set_ylabel(ylabel,fontsize=FSIZE,y=p.yshift)
        # ax.set_xlabel(xlabel,fontsize=FSIZE)
    ax.set_ylim(ymin,ymax)


def set_xticks_v0(ax,x,xlabel):

    # -- ticks --
    x_nm = x[np.where(x < 100)]
    xmin,xmax = x_nm.min().item(),x_nm.max().item()
    xmax += 1

    nuniq = len(np.unique(x))
    xticks = np.arange(nuniq)
    xticks_l = np.linspace(xmin,xmax,nuniq)
    xticklabels = ["%d" % x for x in xticks_l[:-1]]
    xticklabels += ["C"]

    # -- set ticks --
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)

def set_xticks_def(ax,x,xlabel):

    # -- ticks --
    xmin,xmax = x.min().item(),x.max().item()
    nuniq = len(np.unique(x))
    xticks = np.arange(nuniq)
    xticks_l = np.linspace(xmin,xmax,nuniq)
    xticklabels = ["%d" % x for x in xticks_l]

    # -- set ticks --
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)

def get_mesh(X,Y,Z):
    # -- interpolate --
    # lX = np.linspace(min(X), max(X), len(X)+1)
    # lY = np.linspace(min(Y), max(Y), len(Y)+1)
    # mX, mY = np.meshgrid(lX, lY)  # 2D grid for interpolation
    # interp = LinearNDInterpolator(list(zip(X, Y)), Z)
    # # interp = RegularGridInterpolator(list(zip(X, Y)), Z)
    # mZ = interp(mX,mY)

    uX,uY,mZ = get_umesh(X,Y,Z)
    iX = np.arange(len(uX))
    iY = np.arange(len(uY))
    mX, mY = np.meshgrid(iX, iY)  # 2D grid for interpolation

    return mX,mY,mZ

def get_umesh(X,Y,Z):

    # -- lexsort --
    # print(np.c_[X,Y,Z].T)
    # xyz = np.c_[X,Y,Z]
    # xyz = xyz[np.lexsort((xyz[:,0], xyz[:,1], xyz[:,2]))]
    # X,Y,Z = xyz[:,0],xyz[:,1],xyz[:,2]
    # print(np.c_[X,Y,Z].T)

    # -- uniq --
    uX,argsX = np.unique(X,return_inverse=True)
    uY,argsY = np.unique(Y,return_inverse=True)
    args = np.c_[argsX,argsY]
    mZ = np.zeros((len(uX),len(uY)))
    C =  np.zeros((len(uX),len(uY)))
    for a,arg in enumerate(args):
        mZ[arg[0],arg[1]] += Z[a]
        C[arg[0],arg[1]] += 1
    assert np.all(C>=1)
    mZ /= C
    return uX,uY,mZ.T

def prepare_records(records):
    # -- fixed fields --
    pairs = {"stride0":1,"stride1":1,"exact":False}
    for key,val in pairs.items():
        records = records[records[key] == val]

    # -- ngroups --
    cpt = 1./records['ngroups']
    ncg = np.where(records['ngroups']>0,records['ngroups'],records['nchnls'])
    ncg = np.min(np.c_[ncg,records['nchnls']],1)
    cpt = (records['nchnls']-1) // ncg + 1
    records['chnls_pt'] = cpt
    records = records[['ngroups','errors_m','dtime',
                       'neigh_pt','query_pt','chnls_pt','nchnls','rbwd']]
    records.drop_duplicates(inplace=True)
    records = records.rename(columns={"errors_m":"error","dtime":"time"})
    # records = records[~np.isnan(records['errors_m'])]
    # print(records[np.isnan(records['errors_m'])])
    return records

def filter_fixed(records,f0,f1,fixed,rbwd):
    records = records[records['rbwd'] == rbwd]
    for field,fval in fixed.items():
        if field == f0 or field == f1: continue
        records = records[records[field] == fval]
        records.drop_duplicates(inplace=True)
    return records

def plot_colorbar(clims,suffix,in_ax):

    # -- create colorbar --
    if in_ax is None:
        ginfo = {'width_ratios': [1.],'wspace':0.0, 'hspace':0.0,
                 "top":.95,"bottom":0.55,"left":0.05,"right":0.95}
        fig,ax = plt.subplots(1,1,figsize=(5,1),gridspec_kw=ginfo)
    else:
        ax = in_ax

    # -- create colorbar --
    if suffix == "time":
        color = "PiYG"
        norm = mcolors.LogNorm(vmin=clims[0], vmax=clims[1])
    else:
        color = "coolwarm"
        norm = mcolors.Normalize(vmin=clims[0], vmax=clims[1])
    cbar = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                                     norm=norm, extend="both",
                                     cmap=color)

    # -- title --
    FSIZE = 15
    if suffix == "time":
        bname = "Time (seconds)"
    else:
        bname = "Error"
    cbar.set_label(bname,size=FSIZE)

    # -- yicks --
    amin,amax = clims[0],clims[1]
    if suffix == "time":
        ticks = np.logspace(np.log10(amin),np.log10(amax),num=4,endpoint=True)
    else:
        ticks = np.linspace(amin,amax,num=4,endpoint=True)
    ticklabs = ["%2.2f" % x for x in ticks]
    cbar.ax.set_xticks(ticks)
    cbar.ax.set_xticklabels(ticklabs, fontsize=FSIZE)

    # -- save figure --
    if in_ax is None:
        root = SAVE_DIR
        if not root.exists(): root.mkdir(parents=True)
        fn = root / ("colorbar_%s.png" % (suffix))
        plt.savefig(str(fn),dpi=300,transparent=True)
        plt.close("all")

def get_color_bar_lims(records,tgts,fixed,fields):

    # -- init --
    lims = edict()
    lims = {k:[np.inf,-np.inf] for k in tgts}

    # -- over our plotting space --
    mins,maxes = [],[]
    for i0,f0 in enumerate(fields):
        for i1,f1 in enumerate(fields):
            if i0 >= i1: continue
            for rbwd in [True,False]:
                _df = filter_fixed(records,f0,f1,fixed,rbwd)
                for tgt in tgts:
                    tmin = _df[tgt].min()
                    tmax = _df[tgt].max()
                    if tmin < lims[tgt][0]:
                        lims[tgt][0] = tmin
                    if tmax > lims[tgt][1]:
                        lims[tgt][1] = tmax

    return lims
