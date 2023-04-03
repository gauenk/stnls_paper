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
from textwrap import fill
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# -- const --
SAVE_DIR = Path("./output/plots/intro_fig/")


def run(results):

    # -- plot constants --
    FSIZE = 10
    FSIZE_B = 12
    FSIZE_T = 10
    FSIZE_S = 10

    # -- create figure --
    # ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
    #          "top":0.89,"bottom":0.305,"left":0.20,"right":0.57}
    # fig,ax = plt.subplots(figsize=(4,3),gridspec_kw=ginfo)
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.18,"left":0.125,"right":0.98}
    fig,ax = plt.subplots(figsize=(5,2.5),gridspec_kw=ginfo)

    # -- add plots --
    mcolors = {'lidia':'red','colanet':'orange','n3net':'blue'}
    x,y = [],[]
    for mname,res in results.groupby("arch_name"):
        noisy_psnrs = np.stack(res['noisy_psnrs'].to_numpy())
        noisy_psnrs = noisy_psnrs.reshape(2,-1)
        psnrs = np.stack(res['psnrs'].to_numpy()).reshape(2,-1)
        mem_res = np.stack(res['mem_res'].to_numpy()).reshape(2,-1)
        psnrs_m = psnrs.mean(1)
        mem_m = mem_res.mean(1)
        for i in range(2):
            x.append(mem_res[i].mean())
            y.append(psnrs[i].mean())
        mcolor = mcolors[mname]
        lname = res['label_name'].iloc[0]

        # -- arrow --
        our_x = mem_m[0].item()
        their_x = mem_m[1].item()
        dx = our_x - their_x
        our_y = psnrs_m[0].item()
        their_y = psnrs_m[1].item()
        dy = our_y - their_y
        m = dy / dx
        mag_x = 0.05
        mag_y = 0.05
        end_x = our_x - -1*mag_x#*dx
        end_y = our_y - mag_y#*dy
        start_x = their_x + -1*mag_x#*dx
        start_y = their_y + mag_y#*dy
        ax.annotate("", xy=(end_x,end_y), xytext=(start_x,start_y),
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0.1,shrinkB=0.5,
                                    color=mcolor))
        # arrow = mpatches.FancyArrowPatch((a_x,a_y), (a_dx,a_dy), arrowstyle="->",
        #                                  shrinkA=0.,shrinkB=0.,facecolor="k")
        # collection = PatchCollection([arrow])
        # ax.add_collection(collection)
        print(lname,mem_m,psnrs_m)

        # -- add name --
        # ax.annotate("COLA-Net\nOriginal",xy=(start_x,start_y))

        # -- points --
        # print(mname,mem_m)
        # ax.plot([],[],linestyle='-',color=mcolor,label=lname)
        ax.scatter(mem_m[[0]],psnrs_m[[0]],marker='o',color=mcolor)
        ax.scatter(mem_m[[1]],psnrs_m[[1]],marker='x',color=mcolor)

    # -- annotate names --
    # annos = {"colanet":[[.11,28.4],[0.97,27.1]],
    #          "n3net":[[.10,27.9],[.60,27.3]],
    #          "lidia":[[.14,29.45],[.65,28.8]]}
    annos = {"colanet":[[-.00,28.5],[1.4,27.4]],
             "n3net":[[-.02,27.6],[.6,27.3]],
             #"n3net":[[.10,27.9],[.60,27.3]],
             # "lidia":[[.15,29.45],[.65,28.8]]}
             "lidia":[[-.14,29.1],[.65,28.8]]}
    for net,(ours,original) in annos.items():
        res = results[results["arch_name"] == net]
        lname = str(res['label_name'].iloc[0])
        text_ours = "(Ours)"
        text_orig = "(Original)"
        mlen = max(len(lname),len(text_ours))
        # anno_ours = "%s\n%s" % (fill(lname,width=mlen),fill(text_ours,width=mlen))
        # print(mlen,lname,text_ours)
        anno_ours = "%s\n%s" % (lname.center(mlen),text_ours.center(mlen))
        # print(anno_ours)
        mlen = max(len(lname),len(text_orig))
        anno_orig = "%s\n%s" % (fill(lname,width=mlen),fill(text_orig,width=mlen))
        ax.annotate(anno_ours,xy=ours)
        ax.annotate("%s\n%s"%(lname,text_orig),xy=original)


    # -- combine --
    x = np.array(x)
    y = np.array(y)
    print(x)
    print(y)

    # -- legend --
    # plt.plot([], [], 'o', color='k', label="Ours")
    # plt.plot([], [], 'x', color='k', label="Original")

    # -- reset ticks --
    xmin,xmax = x.min().item(),x.max().item()
    ymin,ymax = y.min().item(),y.max().item()
    yticks = np.linspace(ymin,ymax,4)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = np.linspace(xmin,xmax,4)
    xticklabels = ["%1.2f" % x for x in xticks]
    print(yticks,yticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE_T)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE_T)

    # -- limits --
    ax.set_xlim(xmin-.3,xmax+.3)
    ax.set_ylim(ymin-0.25,ymax+0.25)

    # -- axis --
    # title = "Our Space-Time Search Improves Existing Models"
    title = "Our Method Improves PSNR and Reduced GPU Memory"
    # ax.set_title(title,fontsize=FSIZE_B,y=1.02,x=0.70)
    # ax.set_xlabel("Reserved GPU Memory (GB)",fontsize=FSIZE_B,x=0.70)
    # ax.set_ylabel("PSNR (db)",fontsize=FSIZE_B,y=0.50)
    ax.set_title(title,fontsize=FSIZE_B,y=1.0,x=0.5)
    ax.set_xlabel("Reserved GPU Memory (GB)",fontsize=FSIZE,x=0.5)
    ax.set_ylabel("PSNR (dB)",fontsize=FSIZE,y=0.50)

    # -- legend --
    # leg1 = ax.legend(bbox_to_anchor=(0.75,0.99), loc="upper left",fontsize=FSIZE_S,
    #                  title="Method",title_fontsize=FSIZE_S,framealpha=1.,
    #                  edgecolor='k')
    # leg1.get_frame().set_alpha(None)
    # leg1.get_frame().set_facecolor((0, 0, 0, 0.0))


    # -- save --
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)
    fn = SAVE_DIR / "intro_fig.png"
    print(fn)
    plt.savefig(str(fn),dpi=700,transparent=True)

    # # -- create search total --
    # ws = records['ws']
    # wt = records['wt']
    # records['nsearch'] = ws*ws * (2*wt + 1)
    # records['nsearch_t'] = (2*wt + 1)
    # records = records.rename(columns={"deno_mem_res":"mem_res",
    #                                   "bs":"batch_size"})
    # print(records.columns)
    # print(records['mem_res'])

    # # -- plot constants --
    # FSIZE = 18
    # FSIZE_B = 18
    # FSIZE_S = 15

    # # -- init plot --
    # ginfo = {'width_ratios': [1.],'wspace':0.05, 'hspace':0.0,
    #          "top":0.88,"bottom":0.23,"left":0.115,"right":0.97}
    # fig,ax = plt.subplots(1,1,figsize=(8,2.8),gridspec_kw=ginfo)

    # # -- colors to nbwd --
    # lines = ['-','--']
    # colors = {"lidia":"red","colanet":"orange","n3net":"blue"}
    # markers = ["^","x","*"]
    # order = ["lidia","colanet","n3net"]
    # labels = {"lidia":"LIDIA","colanet":"COLA-Net","n3net":"N3Net"}

    # # -- unpack info --
    # ix = 0
    # print(records.columns)
    # for mname in order:
    #     # for mname,df in records.groupby("arch_name"):
    #     df = records[records['arch_name'] == mname]
    #     name = labels[mname]#df['label_name'].iloc[0]
    #     sb = np_col(df['batch_size']).astype(np.int)
    #     order = np.argsort(sb)
    #     sb = sb[order]
    #     xvals = np_col(df['timer_deno'])[order]
    #     yvals = np_col(df['mem_res'])[order]
    #     # print(sb,xvals,yvals)
    #     ax.plot(xvals,yvals,linewidth=2,linestyle='-',
    #             markersize=10,marker=markers[ix],label=name,
    #             color=colors[mname])
    #     ix += 1

    # # -- reset ticks --
    # y = np.stack(records['mem_res'].to_numpy()).ravel()
    # x = np.stack(records['timer_deno'].to_numpy()).ravel()
    # # print(y,x)
    # # print(y.shape,x.shape)
    # xmin,xmax = 0.9*x.min().item(),1.1*x.max().item()
    # ymin,ymax = 0.1*y.min().item(),1.03*y.max().item()
    # yticks = np.linspace(ymin,ymax,5)
    # yticklabels = ["%1.1f" % x for x in yticks]
    # xticks = np.linspace(xmin,xmax,4)
    # xticklabels = ["%d" % x for x in xticks]
    # # for i in range(2):
    # #     if i == 0:
    # #         ax[i].set_yticks(yticks)
    # #         ax[i].set_yticklabels(yticklabels,fontsize=FSIZE)
    # #     else:
    # #         ax[i].set_yticks(yticks)
    # #         ax[i].set_yticklabels([])
    # #     ax[i].set_xticks(xticks)
    # #     ax[i].set_xticklabels(xticklabels,fontsize=FSIZE)
    # #     ax[i].set_ylim(ymin,ymax)

    # # -- set ticks --
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    # ax.set_ylim(ymin,ymax)

    # # -- set labels --
    # ax.set_ylabel("GPU Memory (GB)",fontsize=FSIZE)
    # ax.set_xlabel("Wall-Clock Time (sec)",fontsize=FSIZE)
    # # ax[1].set_xlabel("Num. of Channels",fontsize=FSIZE)

    # # -- format titles --
    # # fig.suptitle("Comparing Execution Times",fontsize=FSIZE_B)
    # ax.set_title("Modulating Memory and Wall-Clock Time",fontsize=FSIZE_B)
    # # ax[1].set_title("Randomized Channels",fontsize=FSIZE_B)

    # # -- legend --
    # leg0 = ax.legend(bbox_to_anchor=(0.668,1.05), loc="upper left",fontsize=FSIZE,
    #                  title="Network",title_fontsize=FSIZE,framealpha=1.,
    #                  edgecolor='k')
    # leg0.get_frame().set_alpha(None)
    # leg0.get_frame().set_facecolor((0, 0, 0, 0.0))

    # # -- save figure --
    # root = SAVE_DIR
    # if not root.exists(): root.mkdir(parents=True)
    # fn = root / "intro_fig.png"
    # plt.savefig(str(fn),dpi=800,transparent=True)
