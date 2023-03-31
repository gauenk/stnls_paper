
"""

Assess the impact of optical flow

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
SAVE_DIR = Path("./output/plots/noisy_flow/")


def run(records):
    records['strred'] *= 100

    # -- init --
    ginfo = {'wspace':0.05, 'hspace':0.17,
             "top":0.93,"bottom":0.10,"left":0.14,"right":0.96}
    fig,axes = plt.subplots(2,1,figsize=(7,6),gridspec_kw=ginfo)

    # -- plot --
    flow_vs_field(records,"psnrs",r"PSNR Difference (dB) $\uparrow$",axes[0],0)
    flow_vs_field(records,"strred",r"ST-RRED Difference $\downarrow$",axes[1],1)

    # -- save --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "noisy_flow.png")
    print("Saving plot at %s" % fn)
    plt.savefig(fn,dpi=500,transparent=True)
    plt.close("all")


def flow_vs_field(records,field,ylabel,in_ax,ax_id):

    # -- input axis --
    axis_s = None

    # -- unpack --
    records[field] = records[field].apply(np.mean)
    df = records[['flow_sigma','sigma','ws',field]]
    # print(df)
    # X = df['flow_sigma'].to_numpy()
    # Y = df['sigma'].to_numpy()
    # Z = df[field].to_numpy()
    # print(X,Y,Z)

    # -- plot constants --
    FSIZE = 16
    FSIZE_B = 16
    FSIZE_S = 15

    # -- init plot --
    if in_ax is None:
        ginfo = {'width_ratios': [1.],'wspace':0.05, 'hspace':0.0,
                 "top":0.89,"bottom":0.18,"left":0.14,"right":0.96}
        fig,ax = plt.subplots(1,1,figsize=(7,4),gridspec_kw=ginfo)
    else:
        ax = in_ax

    # -- create plot --
    Xs,Ys = [],[]
    colors = {15:"yellow",30:"blue",50:"orange"}
    styles = {21:"-",15:"--",7:":"}
    for group,gdf in df.groupby(["sigma","ws"]):
        sigma,ws = group
        X = gdf['flow_sigma'].to_numpy()
        Y = gdf[field].to_numpy()
        args = np.where(X >= 0)[0]
        mX,mY = X[args],Y[args]
        args = np.where(X == -1)[0]
        zX,zY = X[args],Y[args]
        dY = mY - zY
        label = sigma if styles[ws] == "-" else None
        ax.plot(mX,dY,label=label,
                linestyle=styles[ws],
                linewidth=3,
                color=colors[sigma])

        # -- hline --
        # ax.hlines(zY,0,100,linewidth=2,
        #           linestyle=styles[ws],
        #           color=colors[sigma])

        # -- append --
        # print(np.array(list(mX) + list(zX)))
        # print(np.array(list(mY) + list(zY)))
        Xs.append(np.array(list(mX)))
        Ys.append(np.array(list(dY)))

        # -- annotate "No Flow" --
        # start_x = 0
        # start_y = zY.item()-0.5
        # end_x = start_x + 0.05
        # end_y = start_y + 0.05
        # ax.annotate("No Flow", xy=(end_x,end_y), xytext=(start_x,start_y),
        #             fontsize=13)

        # -- annotate "sigma" --
        # start_x = 0
        # start_y = zY.item()+0.25
        # end_x = start_x + 0.05
        # end_y = start_y + 0.05
        # ax.annotate("Video Noise to Denoise ($\sigma^2 = %d$)" % sigma,
        #             xy=(end_x,end_y),
        #             xytext=(start_x,start_y),
        #             fontsize=12,color=colors[sigma])

        # -- annotate "ws" --
        if sigma == 30 and field == "psnrs":
            start_x = 0
            end_x = start_x + 0.05
            start_y = dY[0].item()+0.05
            end_y = start_y + 0.10
            ax.annotate("ws = %d" % ws,
                        xy=(end_x,end_y),
                        xytext=(start_x,start_y),
                        fontsize=14)

        if field == "strred":
            if sigma == 50 and ws < 21:
                start_x = 0
                end_x = start_x + 0.05
                if ws == 7:
                    start_y = dY[0].item()+0.3
                else:
                    start_y = dY[0].item()+0.2
                end_y = start_y + 0.10
                ax.annotate("ws = %d" % ws,
                            xy=(end_x,end_y),
                            xytext=(start_x,start_y),
                            fontsize=14)
            elif sigma == 30 and ws == 21:
                start_x = 0
                end_x = start_x + 0.05
                start_y = dY[0].item()+0.10
                end_y = start_y + 0.10
                ax.annotate("ws = %d" % ws,
                            xy=(end_x,end_y),
                            xytext=(start_x,start_y),
                            fontsize=14)

        # ax.annotate("hi", xy=(end_x,end_y), xytext=(start_x,start_y),
        #             arrowprops=dict(arrowstyle="->",
        #                             shrinkA=0.1,shrinkB=0.5,
        #                             color=colors[sigma]))

    # -- get ticks --
    x = np.array(Xs)#df['flow_sigma']#.to_numpy().ravel()
    y = np.array(Ys)#df[field]#.to_numpy().ravel()
    xmin,xmax = x.min().item(),x.max().item()
    # ymin,ymax = y.min().item()-0.25,y.max().item()+0.25
    ymin,ymax = y.min().item(),y.max().item()
    delta = .15*(ymax - ymin)
    ymin,ymax = ymin-delta,ymax+delta
    ymin = min(ymin,0)
    if field == "strred":
        ymax = min(0,ymax)
    yticks = np.linspace(ymin,ymax,5)
    yticklabels = ["%1.2f" % x for x in yticks]
    xticks = np.unique(Xs)#np.linspace(xmin,xmax,5)
    xticks = list(xticks)
    xticklabels = ["%d" % x for x in xticks]
    if ax_id == 0:
        xticks = []
        xticklabels = []

    # -- reset axis --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    ax.set_ylim(ymin,ymax)

    # -- legend --
    # ax.plot([],[],label=30,color=colors[30])
    # ax.plot([],[],label=50,color=colors[50])
    loc = (0.625,0.75) if field == "psnrs" else (0.625,0.3)
    if ax_id == 0:
        leg = ax.legend(title="$\sigma^2$",loc="upper left",
                        bbox_to_anchor=loc,framealpha=1.,
                        edgecolor='k',ncols=2,
                        title_fontsize=13,fontsize=13)
        leg.get_frame().set_alpha(None)
        leg.get_frame().set_facecolor((0, 0, 1., 0))

    # -- labels --
    y_shift = 1.05
    x_shift = 0.5
    if ax_id == 0:
        full_title = "Noisy Optical Flow Estimates Improve Denoising Quality"
        ax.set_title(full_title,fontsize=FSIZE_B,
                     y=y_shift,x=x_shift)
    # ax.set_ylabel("PSNR Difference (dB)",fontsize=FSIZE)
    ax.set_ylabel(ylabel,fontsize=FSIZE)
    if ax_id == 1:
        ax.set_xlabel("Noise When Estimating Optical Flow ($\sigma_f^2$)",
                      fontsize=FSIZE)

    # -- save figure --
    if in_ax is None:
        root = SAVE_DIR
        if not root.exists(): root.mkdir(parents=True)
        fn = str(root / ("noisy_flow_%s.png" % field))
        print("Saving plot at %s" % fn)
        plt.savefig(fn,dpi=500,transparent=True)
        plt.close("all")

