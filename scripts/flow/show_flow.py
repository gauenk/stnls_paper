
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- vision --
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF

# -- data io --
import data_hub

# -- management --
from pathlib import Path
from easydict import EasyDict as edict

def main():

    #
    # -- params --
    #

    # -- save info --
    save_dir = Path("./output/plots/show_flow/")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # -- data args --
    nskip,nstart = 3,45
    data_cfg = edict()
    data_cfg.dname = "set8"
    data_cfg.vid_name = "snowboard"
    data_cfg.nframes = 3*nskip+nstart
    data_cfg.sigma = 30.

    # -- cropping args --
    crop_cfg = edict()
    crop_cfg.size = 96
    crop_cfg.crops = [[300,385],[310,445],[290,490]]

    # -- highlight --
    lit_cfg = edict()
    lit_cfg.size = 256
    lit_cfg.crop = [256,335]


    # -- load data --
    vname = data_cfg.vid_name
    data,loaders = data_hub.sets.load(data_cfg)
    indices = [idx for idx,key in enumerate(data.te.groups) if vname in key]
    vid = data.te[indices[0]]['clean'][nstart::nskip] # get video
    nframes = vid.shape[0]

    #
    # -- higlight --
    #

    lits = []
    for t,frame in enumerate(vid):

        # -- lit frame --
        size = lit_cfg.size
        loc_c = lit_cfg.crop
        lit_t = TF.crop(frame,loc_c[0],loc_c[1],size,size)
        lit_t = lit_t.type(th.uint8)
        is_mid = t == nframes//2

        # -- init mask --
        c,h,w = lit_t.shape
        masks = th.zeros((2,h,w)).bool()
        # print("lit_t.shape: ",lit_t.shape)

        # -- proposed --
        size = crop_cfg.size
        loc_t = crop_cfg.crops[t]
        loc_t = [loc_t[0]-loc_c[0],loc_t[1]-loc_c[1]]
        is_mid_mod = size//2 if is_mid else 0 # lower-half if mid
        h_slice = slice(loc_t[0]+is_mid_mod,loc_t[0]+size)
        w_slice = slice(loc_t[1],loc_t[1]+size)
        masks[0,h_slice,w_slice] = 1

        # -- center --
        size = crop_cfg.size
        loc_mid = crop_cfg.crops[nframes//2]
        loc_mid = [loc_mid[0]-loc_c[0],loc_mid[1]-loc_c[1]]
        is_mid_mod = size//2 if is_mid else size # upper-half if mid
        h_slice = slice(loc_mid[0],loc_mid[0]+is_mid_mod)
        w_slice = slice(loc_mid[1],loc_mid[1]+size)
        masks[1,h_slice,w_slice] = 1

        # -- raw --
        # save_image(lit_t/255.,str(save_dir / ("raw_%d.png" % t)))

        # -- draw --
        lit_t = draw_segmentation_masks(lit_t,masks[[0]],alpha=0.4,colors="orange")
        lit_t = draw_segmentation_masks(lit_t,masks[[1]],alpha=0.4,colors="blue")

        # -- append --
        lits.append(lit_t)

    # -- save lit images --
    lits = th.stack(lits).float()/255.
    # print("lits.shape: ",lits.shape)
    fn = str(save_dir / "highlights.png")
    print("Saved figure %s" % fn)
    save_image(lits,fn)

    # -- save each --
    for i in range(len(lits)):
        fn = str(save_dir / ("highlight_%d.png" % i))
        print("Saved figure %s" % fn)
        save_image(lits[i],fn)


    #
    # -- collect crops --
    #

    centers,proposed = [],[]
    for t,frame in enumerate(vid):
        size = crop_cfg.size
        loc_t = crop_cfg.crops[t]
        prop_t = TF.crop(frame,loc_t[0],loc_t[1],size,size).type(th.uint8)
        masks_t = th.ones_like(prop_t).type(th.bool)[:1]
        prop_t = draw_segmentation_masks(prop_t,masks_t,alpha=0.2,colors="orange")
        loc_mid = crop_cfg.crops[nframes//2]
        center_t = TF.crop(frame,loc_mid[0],loc_mid[1],size,size).type(th.uint8)
        masks_t = th.ones_like(center_t).type(th.bool)[:1]
        center_t = draw_segmentation_masks(center_t,masks_t,alpha=0.2,colors="blue")
        proposed.append(prop_t)
        centers.append(center_t)
    proposed = th.stack(proposed)
    centers = th.stack(centers)
    # print(proposed.shape)
    # print(centers.shape)

    # -- save grid --
    t = proposed.shape[0]
    agg = th.cat([proposed,centers])
    grid = make_grid(agg,nrow=t)/255.
    fn = str(save_dir / "zoomed_grid.png")
    print("Saved figure %s" % fn)
    save_image(grid,fn)

    # -- save each --
    # for t in range(t):
    #     prop_t = proposed[t][None,:]/255.
    #     save_image(prop_t,str(save_dir / ("dynamic_%d.png"%t)))
    #     cent_t = centers[t][None,:]/255.
    #     save_image(cent_t,str(save_dir / ("static_%d.png"%t)))

if __name__ == "__main__":
    main()
