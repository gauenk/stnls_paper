import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():

    q_list = [3,5]
    m_list = ['x','+']
    wgrid = np.arange(3,13+1)[::2]

    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.13,"left":.18,"right":0.95}
    fig,ax = plt.subplots(figsize=(4.5,3.5),gridspec_kw=ginfo,dpi=200)

    for q,m in zip(q_list,m_list):
        nls = (q+wgrid-1)**2
        snls = q*(wgrid)**2
        print(nls)
        print(snls)
        ax.plot(wgrid,nls,'%s--'%m,label="Overlapping (Q=%d)"%q,color="red")
        ax.plot(wgrid,snls,'%s-'%m,label="Non-Overlapping (Q=%d)"%q,color="blue")

    ax.legend(framealpha=0.)
    ax.set_title("Tiling is Not Practical for a Shifted-NLS")

    ax.set_xticks(wgrid)
    ax.set_xticklabels(["%d" % w for w in wgrid])

    ax.set_xlabel("Spatial Window Size (W)")
    ax.set_ylabel("Number of Reads from Global Memory")

    plt.savefig("patchdb_growth.png",transparent=True)


if __name__ == "__main__":
    main()
