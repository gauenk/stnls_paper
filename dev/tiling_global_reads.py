import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path

def main():

    q_list = [3]
    m_list = ['x','+']
    wgrid = np.arange(3,13+1)[::2]

    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.20,"left":.18,"right":0.99}
    fig,ax = plt.subplots(figsize=(3.5,2.25),gridspec_kw=ginfo,dpi=200)

    ix = 0
    for q,m in zip(q_list,m_list):
        nls = (q+wgrid-1)
        snls = q*(wgrid)**2
        print("q: ",q)
        print("nls: ",nls)
        print("snls: ",snls)
        fmt = "-" if ix == 0 else "--"
        ax.plot(wgrid,nls,'o'+fmt,label="Overlapping (Space)",color="red")
        ax.plot(wgrid,snls,'x'+fmt,label="Non-Overlapping\n(Space-Time)",color="blue")
        ix += 1

    ax.legend(framealpha=0.)
    ax.set_title("Space-Time Reads More than Space-Only",fontsize=11,x=0.4)

    ax.set_xticks(wgrid)
    ax.set_xticklabels(["%d" % w for w in wgrid])

    ax.set_xlabel("Spatial Window Size (W)")
    ax.set_ylabel("Reads from Global Memory",y=0.40)


    root = Path("output/figures/")
    if not(root.exists()):
        root.mkdir(parents=True)
    plt.savefig(root/"tiling_global_reads.png",transparent=True)


if __name__ == "__main__":
    main()
