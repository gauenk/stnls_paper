"""

A script to highlight regions from denoised examples

"""


# -- data mng --
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(depth=5,indent=8)
import copy
dcopy = copy.deepcopy

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat
import cache_io

# -- vision --
import cv2
from PIL import Image
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# -- dev basics --
from dev_basics.utils.misc import optional
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
    elif vid_name == "giant-slalom":
        regions = []
        regions.append('3_140_690_200_750')
        regions.append('5_140_690_200_750')
        regions.append('6_140_690_200_750')
        regions.append('7_140_690_200_750')
        regions.append('8_140_690_200_750')
        regions.append('9_140_690_200_750')
    elif vid_name == "rafting":
        regions = []
        regions.append('9_408_512_472_576')
        regions.append('9_0_0_-1_-1')
    elif "tandem" in vid_name:
        regions = []
        regions.append('3_356_338_420_402')
        regions.append('3_0_0_-1_-1')
    elif vid_name == "tennis-vest":
        regions = []
        regions.append('4_130_338_182_390')
        regions.append('4_130_338_182_390')
        # regions.append('4_0_0_-1_-1')
    elif vid_name == "car-race":
        regions = []
        # regions.append('1_288_120_-1_365')
        # regions.append('2_288_120_-1_365')

        # regions.append('3_328_240_440_365')
        # regions.append('4_328_120_440_365')

        regions.append('7_357_365_407_415')
        regions.append('8_357_365_407_415')
        regions.append('9_357_365_407_415')
    elif vid_name == "horsejump-stick":
        regions = []
        regions.append('9_110_300_240_430')
        pass
    else:
        print(f"Please select a real region for {vid_name}!")
        regions = []
        regions.append('1_0_0_-1_-1')
        regions.append('2_0_0_-1_-1')
        regions.append('3_0_0_-1_-1')
        regions.append('6_0_0_-1_-1')
        regions.append('7_0_0_-1_-1')
        regions.append('8_0_0_-1_-1')
        regions.append('9_0_0_-1_-1')
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

def increase_brightness(img, value=30):

    # device = img.device
    # img = img.cpu().numpy()
    # img = rearrange(img,'c h w -> h w c')*1.
    img = img*1.
    img /= img.max()
    img *= 255
    img = np.clip(img,0,255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value
    v = v*1.
    v -= v.min()
    v /= v.max()
    v *= 255
    v = v.astype(np.uint8)


    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    # img = th.from_numpy(img).to(device)
    # img = rearrange(img,'h w c -> c h w')
    # img /= img.max()

    return img

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
        if vid.max() < 200: # guess at normalization
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
        vid_t = increase_brightness(vid[ti])
        # vid_t = vid[ti]
        vid_t = Image.fromarray(vid_t)
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
    return vid

def save_examples(vids,root,regions,psnrs):

    # -- show zoomed regions on larger vid --
    # save_highlight(vids.clean,regions,root,"clean_highlight")

    # -- save zoomed regions --
    for vid_cat,vid in vids.items():
        print(vid_cat,vid.min(),vid.max())
        save_regions(vid,regions,root,vid_cat)

def load_denoised(cfg,nframes,frame_start):
    saved_dir = optional(cfg,"saved_dir","output/saved_examples/%s/"%str(cfg.arch_name))
    vdir = Path(saved_dir) / cfg.uuid
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
    dcfg = dcopy(cfg)
    data,loaders = data_hub.sets.load(dcfg)
    index = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                   cfg.frame_start,cfg.frame_end)[0]
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

def get_psnrs(vids,regions):
    psnrs = edict()
    psnrs.noisy = compute_psnrs(vids.clean,vids.noisy,255)
    psnrs.ours = compute_psnrs(vids.clean,vids.deno_ours,255)
    psnrs.orig = compute_psnrs(vids.clean,vids.deno_orig,255)
    return psnrs

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

def get_topk_regions(vids,K):
    ours = th.abs(vids.clean - vids.deno_ours)
    orig = th.abs(vids.clean - vids.deno_orig)
    delta = orig - ours
    print(delta.shape)

    T,C,H,W = vids.clean.shape
    windows = [64,128]
    def get_grid(ws,N):
        locs_h = th.linspace(ws//2,int(H-ws//2),N)
        locs_w = th.linspace(ws//2,int(W-ws//2),N)
        grid_y, grid_x = th.meshgrid(locs_h,locs_w)
        grid = th.stack((grid_y, grid_x),2).long()  # H(x), W(y), 2
        grid = grid.view(-1,2)
        return grid

    vals = []
    regions = []
    for t in range(T):
        for ws in windows:
            grid = get_grid(ws,8)
            for loc in grid:
                print(t,ws,loc)
                sH,sW = max(loc[0]-ws//2,0),max(loc[1]-ws//2,0)
                eH,eW = sH+ws,sW+ws
                # val_d = th.mean(delta[t,:,sH:eH,sW:eW]).item()
                val_ours = th.mean(ours[t,:,sH:eH,sW:eW]).item()
                val_orig = th.mean(orig[t,:,sH:eH,sW:eW]).item()
                vals.append(val_orig/(val_ours+1e-8))
                reg_loc = "%d_%d_%d_%d_%d" % (t,sH,sW,eH,eW)
                regions.append(reg_loc)
    vals = th.tensor(vals)
    topk = th.topk(vals,K,largest=True)
    regions = [regions[k] for k in topk.indices]
    print(regions,topk.values)
    return regions

def run(cfg,cfg_orig,subdir,K=3,nframes=-1,frame_start=0):

    # -- save info --
    save_dir = Path("./output/deno_examples/") / subdir
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    root = SAVE_DIR/subdir/cfg.vid_name
    # print(root)
    # exit()

    # -- load videos --
    vids = prepare_vids(cfg,cfg_orig,nframes,frame_start)

    # -- get region --
    vid_name = cfg['vid_name']
    regions = get_regions(vid_name)
    # regions = get_topk_regions(vids,K)
    print(vid_name)

    # -- load psnrs --
    # psnrs = prepare_psnrs(cfg,cfg_orig,vids,regions,nframes,frame_start)
    psnrs = get_psnrs(vids,regions)
    # psnrs = np.array([-1,]*10)
    print(psnrs)

    # -- save region --
    save_examples(vids,root,regions,psnrs)

    # -- save delta --
    # save_deltas(vids,root)

def get_paired_deno():
    exp_fn = edict()
    exp_fn.rvrt = "exps/trte_rvrt/test.cfg"
    exp_fn.nlnet = "exps/trte_nlnet/test.cfg"

    cache = edict()
    cache.rvrt = cache_io.ExpCache(".cache_io/trte_rvrt/test")
    cache.nlnet = cache_io.ExpCache(".cache_io/trte_nlnet/test")

    cfgs = edict()
    cfgs.rvrt = cache_io.read_test_config.run(exp_fn.rvrt)
    cfgs.nlnet = cache_io.read_test_config.run(exp_fn.nlnet)

    # vids = ["rafting","tandem","tennis-vest"]
    # vids = ["giant-slalom"]#,"man-bike","guitar-violin"]
    # vids = ["car-race"]
    # vids = ["tennis-vest"]
    # vids = ["monkeys-trees"]
    vids = ["horsejump-stick"]
    paired_cfgs = {}
    ix = 0
    for rvrt in cfgs.rvrt:
        # ix += 1
        # if ix < 5: continue
        # print(rvrt.vid_name)
        # if not("salsa" in rvrt.vid_name): continue
        if rvrt.sigma != 50: continue
        name = rvrt.vid_name
        if not(name in vids): continue
        nlnet = [nlnet for nlnet in cfgs.nlnet if nlnet.vid_name == name]
        nlnet = [n for n in nlnet if n.sigma == 50][0]
        rvrt.uuid = cache.rvrt.get_uuid(rvrt)
        nlnet.uuid = cache.nlnet.get_uuid(nlnet)
        # print(rvrt.vid_name,rvrt.uuid,nlnet.uuid)
        print(rvrt.vid_name,rvrt.uuid,nlnet.uuid)
        paired_cfgs[name] = {"orig":rvrt,"ours":nlnet}
        # pp.pprint(cfg)
        # res = cache.load_exp(cfg) # load result
        # print(res.keys())
        # paired_cfgs[name][cls] = res
        # break
        # if ix > 5: break
    return paired_cfgs

def get_paired_rvrt():
    exp_fn = "exps/trte_rvrt/test.cfg"
    cache = cache_io.ExpCache(".cache_io/trte_rvrt/test")
    raw_cfgs = cache_io.read_test_config.run(exp_fn)
    paired_cfgs = {}
    vids = ["rafting","tandem","tennis-vest"]
    for cfg in raw_cfgs:
        name = cfg.vid_name
        if not(name in vids): continue
        cls = "ours" if cfg.offset_type == "search" else "orig"
        if not(name in paired_cfgs):
            paired_cfgs[name] = {}
        cfg.uuid = cache.get_uuid(cfg) # load result
        paired_cfgs[name][cls] = cfg
    return paired_cfgs

def main():

    # -- create denos --
    # paired_cfgs = get_paired_rvrt()
    subdir = "scaled"
    paired_cfgs = get_paired_deno()
    for vid_name in paired_cfgs.keys():
        ours = paired_cfgs[vid_name]['ours']
        orig = paired_cfgs[vid_name]['orig']
        run(ours,orig,subdir,K=10,nframes=10,frame_start=0)

if __name__ == "__main__":
    main()
