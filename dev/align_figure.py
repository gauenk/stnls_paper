
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse


def main():

    psnrs = np.array([75.4204225,  72.33091522, 78.66552541, 80.55617397])
    times = np.array([.09070,      0.02919,     0.4037,      4.542])
    mems = np.array([0.117,      0.047,     0.047,      0.047])

    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.13,"left":.18,"right":0.95}
    fig,ax = plt.subplots(figsize=(4.5,3.5),gridspec_kw=ginfo,dpi=200)

    ax.scatter(times,psnrs,s=1000*mems)
    ax.set_xscale('log')

    plt.savefig("align_figure.png",transparent=True)


if __name__ == "__main__":
    main()
