"""

A script to highlight regions from denoised examples

"""


# -- data mng --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- vision --
from PIL import Image
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# -- dev basics --
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

# -- data io --
import data_hub

# -- management --
from pathlib import Path
from easydict import EasyDict as edict

# -- plotting --
from matplotlib import pyplot as plt

SAVE_DIR = Path("./output/deno_examples/")

def get_regions(vid_name):
    if vid_name == "sunflower":
        regions = []
        # regions.append('3_0_0_-1_-1')
        # regions.append('3_128_384_384_640')
        # regions = ["%d_0_256_512_768" % t for t in range(84)]
        # regions = ["%d_128_384_384_640" % t for t in range(84)]
        # regions = ["%d_192_448_320_576" % t for t in range(84)]
        # regions = ["%d_224_480_288_544" % t for t in range(84)]
        # regions = ["%d_192_448_320_576" % t for t in range(45,55)]
        regions = ["%d_224_416_288_480" % t for t in range(45,55)]
    elif vid_name == "tractor":
        regions = []
        regions.append('0_0_0_-1_-1')
        regions.append('0_200_580_264_644')
    elif vid_name == "touchdown":
        regions = []
        regions.append('0_0_0_-1_-1')
        regions.append('0_178_128_434_384')
    else:
        print(f"Please select a real region for {vid_name}!")
        regions = []
        regions.append('0_0_0_-1_-1')
        regions.append('0_256_500_384_628')
    return regions

def vid_region(vid,region):
    t,h0,w0,h1,w1 = [int(r) for r in region.split("_")]
    hslice = slice(h0,h1)
    wslice = slice(w0,w1)
    return vid[[t],:,hslice,wslice]

def highlight_mask(vid,region):
    t,h0,w0,h1,w1 = [int(r) for r in region.split("_")]
    hslice = slice(h0,h1)
    wslice = slice(w0,w1)
    mask = th.zeros_like(vid).bool()
    mask[[t],:,hslice,wslice] = 1
    return mask

def higlight_slice(vid,region,color,alpha=0.4):
    # -- highlight selected region --
    mask = highlight_mask(vid,region)
    t = mask.shape[0]
    for ti in range(t):
        vid[ti] = draw_segmentation_masks(vid[ti],mask[ti],alpha=alpha,colors=color)
    return vid

def get_vid_from_data(data,vid_name):
    groups = data.groups
    indices = [i for i,g in enumerate(groups) if vid_name in g]
    assert len(indices) == 1
    vid = data[indices[0]]['clean']
    return vid

def save_video(vid,root,fname):

    # -- format --
    if th.is_tensor(vid):
        vid = vid.cpu().numpy()
    if vid.dtype != np.uint8:
        vid /= vid.max()
        vid *= 255
        vid = np.clip(vid,0.,255.)
        vid = vid.astype(np.uint8)

    # -- create root --
    if not root.exists():
        root.mkdir(parents=True)

    # -- save burst --
    vid = rearrange(vid,'t c h w -> t h w c')
    t = vid.shape[0]
    for ti in range(t):
        vid_t = Image.fromarray(vid[ti].squeeze())
        fname_t = fname + "_%d.png" % ti
        vid_t.save(str(root / fname_t))

def save_regions(vid,regions,root,fname):
    for r,region in enumerate(regions):
        vid_r = vid_region(vid,region)
        save_video(vid_r,root,fname+"_%d"%r)

def save_highlight(vid,regions,root,fname):
    vid = (vid).type(th.uint8)
    colors = ["red","yellow","blue"]
    for r,region in enumerate(regions):
        if r == 0: continue
        cidx = r % len(colors)
        color = colors[cidx]
        vid = higlight_slice(vid,region,color,alpha=0.4)
    save_video(vid,root,fname)

def load_from_results(df):
    home = Path(df['home_path'].iloc[0])
    paths = df['deno_fns'].iloc[0][0]
    vid = []
    for path in paths:
        path_f = home / path
        img = Image.open(path_f).convert("RGB")
        img = np.array(img)
        img = rearrange(img,'h w c -> c h w')
        vid.append(img)
    vid = np.stack(vid)
    print("vid.shape: ",vid.shape)
    return vid

def save_examples(vids,root,regions,psnrs):

    # -- show zoomed regions on larger vid --
    save_highlight(vids.clean,regions,root,"clean_highlight")

    # -- save zoomed regions --
    for vid_cat,vid in vids.items():
        save_regions(vid,regions,root,vid_cat)

def load_denoised(cfg,nframes,frame_start):
    vdir = Path(cfg.saved_dir) / cfg.uuid
    fns = sorted(list(vdir.iterdir()))
    if nframes > 0:
        fns = fns[frame_start:frame_start+nframes]
    vid = []
    for fn in fns:
        img = np.array(Image.open(str(fn)))
        img = th.tensor(img.transpose(2,0,1))
        vid.append(img)
    vid = th.stack(vid)
    return vid


def prepare_vids(cfg,cfg_orig,nframes,frame_start):

    # -- enpoints --
    if nframes > 0:
        fs,fe = frame_start,frame_start + nframes
    else:
        fs,fe = 0,-1

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    index = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                   cfg.frame_start,cfg.nframes)[0]
    clean = data[cfg.dset][index]['clean'][fs:fe]
    noisy = data[cfg.dset][index]['noisy'][fs:fe]

    # -- format noisy --
    noisy = (th.clamp(noisy,0.,255.)).type(th.uint8)

    # -- load denoised --
    deno_ours = load_denoised(cfg,nframes,frame_start)
    deno_orig = load_denoised(cfg_orig,nframes,frame_start)

    # -- vid compact --
    vids = edict()
    vids.clean = clean
    vids.noisy = noisy
    vids.deno_ours = deno_ours
    vids.deno_orig = deno_orig

    for key in vids:
        print(key,vids[key].shape)
    return vids

def prepare_psnrs(cfg_ours,cfg_orig,vids,regions,nframes,frame_start):

    # -- endpoints --
    if nframes > 0:
        fs,fe = frame_start,frame_start + nframes
    else:
        fs,fe = 0,-1

    # -- unpack "t" list --
    print(regions)
    tlist = [int(r.split("_")[0]) for r in regions]

    # -- noisy psnrs --
    npsnrs = compute_psnrs(vids.noisy,vids.clean,255)
    print(cfg_ours['psnrs'])

    # -- psnrs compact --
    psnrs = edict()
    psnrs.clean = [0. for t in tlist]
    psnrs.noisy = [npsnrs[t] for t in tlist]
    psnrs.deno_ours = [cfg_ours['psnrs'][0][t] for t in tlist]
    psnrs.deno_orig = [cfg_orig['psnrs'][0][t] for t in tlist]
    return psnrs

# def save_deltas(vids,root):
#     vids.deno_ours - vids.clean
#     vids.deno_orig - vids.clean

def run(cfg,cfg_orig,nframes=-1,frame_start=0):

    # -- save info --
    arch_name = str(cfg['arch_name'])
    save_dir = Path("./output/deno_examples/") / arch_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    root = SAVE_DIR/arch_name/cfg.vid_name
    print(root)

    # -- load videos --
    vids = prepare_vids(cfg,cfg_orig,nframes,frame_start)

    # -- get region --
    vid_name = cfg['vid_name']
    regions = get_regions(vid_name)
    print(vid_name)

    # -- load psnrs --
    psnrs = prepare_psnrs(cfg,cfg_orig,vids,regions,nframes,frame_start)
    print(psnrs)

    # -- save region --
    save_examples(vids,root,regions,psnrs)

    # -- save delta --
    # save_deltas(vids,root)

def main():
    pass

if __name__ == "__main__":
    main()
