"""

Format the directory structure for a specific set of results using symbolic links.

"""

import os,tqdm
import data_hub
import numpy as np
import torch as th
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
import cache_io
import dev_basics.exps as dev_exps
from torchvision.utils import make_grid
from dev_basics.utils import vid_io
from PIL import Image
from einops import rearrange

def get_gif_dir(base,version,cfg):
    if version == 0:
        subdir = Path("%s_%d" % (cfg.dname,cfg.sigma))
        root = base / "gif" / ("version_%d" % version) / cfg.arch_name
        root /= subdir
    else:
        raise ValueError(f"Uknown verison [{version}]")
    if not root.exists():
        root.mkdir(parents=True)
    # if cfg.wt == 0: gid = "orig"
    # else: gid = "ours"
    fn = root / cfg.vid_name #("%s.gif" % (cfg.vid_name))
    return fn

def get_sym_root(base,version,cfg):
    if cfg.wt == 0: gid = "orig"
    else: gid = "ours"
    if version == 0:
        subdir = Path("%s_%d" % (cfg.dname,cfg.sigma))
        subdir /= "%s/%s_%s" % (cfg.arch_name,cfg.vid_name,gid)
    elif verion == 1:
        subdir = Path("%s_%d" % (cfg.dname,cfg.sigma))
        subdir /= "%s/%s/%s" % (cfg.arch_name,cfg.vid_name,gid)
    else:
        raise ValueError(f"Uknown verison [{version}]")
    root = base / "img" / ("version_%d" % version) / subdir
    if not root.exists():
        root.mkdir(parents=True)
    return root

def create_links_res(base,version,cfg,uuid,result):
    sym_root = get_sym_root(base,version,cfg)
    if result is None: return
    fns = result['deno_fns'][0]
    for fn in fns:
        fn = Path(fn)
        if (not fn.exists()) or not(fn.suffix == ".png"):
            break
        src_fn = str(fn.resolve())
        dest_fn = sym_root / fn.name
        if dest_fn.exists(): continue
        os.symlink(src_fn,str(dest_fn))

def vid_merge(*vids,nrow=2):
    vid = []
    for frames in zip(*vids):
        grid = make_grid(list(frames),nrow=nrow)
        grid = grid.numpy().astype(np.uint8)
        grid = rearrange(grid,'c h w -> h w c')
        grid = Image.fromarray(grid)
        vid.append(grid)
    return vid

def read_nc_pair(cfg):
    data,loaders = data_hub.sets.load(cfg)
    index = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                   cfg.frame_start,cfg.frame_end)[0]
    clean = data[cfg.dset][index]['clean']
    noisy = data[cfg.dset][index]['noisy']
    return clean,noisy

def region_clip(cfg,*args):
    fps = 1.
    regions = [None,[200,456,200,456]]
    if cfg.vid_name == "sunflower":
        regions += [[210,298,406,490]]
    elif cfg.vid_name == "tractor":
        regions += [[180,336,100,256],[200,264,580,644]]
    elif cfg.vid_name == "park_joy":
        regions += [[220,314,220,314]]
    elif cfg.vid_name == "bike-packing":
        regions += [[100,190,800,890]]
    elif cfg.vid_name == "judo":
        regions += [[0,200,300,500]]
    elif cfg.vid_name == "breakdance":
        regions += [[150,270,0,100],[175,295,100,200],[150,250,525,650]]
    elif cfg.vid_name == "scooter-black":
        fps = 0.3
        regions += [[0,96,854-96,854]]
    elif cfg.vid_name == "cows":
        fps = 1
        regions += [[100,356,400,656],[288,384,384,480]]
    return cropped_at(regions,*args),fps

def cropped_at(regions,*vids):
    crops = []
    for region in regions:
        if region is None:
            _vids = [th.clamp(vid,0.,255.) for vid in vids]
            crops.append(_vids)
        else:
            h0,h1,w0,w1 = region
            _crops = []
            for vid in vids:
                vid = th.clamp(vid,0.,255.)
                _crops.append(vid[...,h0:h1,w0:w1])
            crops.append(_crops)
    return crops

def create_gif_res(base,version,cfg_p,uuid_p,result_p,overwrite=False):

    # -- edge --
    if result_p[0] is None: return
    if result_p[0]['deno_fns'][0][0] == "": return

    # -- filename --
    gif_dir = get_gif_dir(base,version,cfg_p[0])
    if not gif_dir.exists():
        gif_dir.mkdir(parents=True)

    # -- read noisy/clean --
    clean,noisy = read_nc_pair(cfg_p[0])

    # -- read denoised --
    fns0 = result_p[0]['deno_fns'][0]
    vid0 = vid_io.read_files(fns0)
    fns1 = result_p[1]['deno_fns'][0]
    vid1 = vid_io.read_files(fns1)

    # -- merge & save --
    regions,fps = region_clip(cfg_p[0],clean,noisy,vid0,vid1)
    print(len(regions),"regions")
    for r_index,region in enumerate(regions):

        # -- optional skip --
        gif_fn = gif_dir / ("%d.gif" % r_index)
        print(gif_fn.exists(),overwrite)
        if gif_fn.exists() and not(overwrite):
            continue
        print(gif_fn)

        # -- merge --
        clean,noisy,vid0,vid1 = region
        vid = vid_merge(noisy,vid1,vid0,clean,nrow=4)

        # -- save gif --
        dur = int(len(vid)/fps)
        vid[0].save(str(gif_fn), format='GIF',
                    append_images=vid[1:],
                    save_all=True,duration=dur, loop=0)

def create_links(base,version,src,exps):
    uuids,configs,results = src.load_raw_configs(exps,True)
    N = len(results)
    for cfg,uuid,result in tqdm.tqdm(zip(configs,uuids,results),total=N):
        create_links_res(base,version,cfg,uuid,result)

def create_gifs(base,version,src,exps,ow_gifs):
    uuids,configs,results = src.load_raw_configs(exps,True)
    uuids,configs,results = create_pairs(uuids,configs,results)
    N = len(results)
    for cfg_p,uuid_p,result_p in tqdm.tqdm(zip(configs,uuids,results),total=N):
        create_gif_res(base,version,cfg_p,uuid_p,result_p,ow_gifs)

def create_crops(base,version,src,exps,ow_gifs):
    uuids,configs,results = src.load_raw_configs(exps,True)
    uuids,configs,results = create_pairs(uuids,configs,results)
    N = len(results)
    for cfg_p,uuid_p,result_p in tqdm.tqdm(zip(configs,uuids,results),total=N):
        create_crop_res(base,version,cfg_p,uuid_p,result_p,ow_gifs)

def create_crop_res(base,version,cfg_p,uuid_p,result_p,overwrite=False):

    # -- edge --
    replace_uuid(cfg_p,uuid_p,result_p)
    if result_p[0] is None: return
    if result_p[0]['deno_fns'][0][0] == "": return

    # -- filename --
    gif_dir = get_gif_dir(base,version,cfg_p[0])
    if not gif_dir.exists():
        gif_dir.mkdir(parents=True)

    # -- read noisy/clean --
    clean,noisy = read_nc_pair(cfg_p[0])

    # -- read denoised --
    fns0 = result_p[0]['deno_fns'][0]
    vid0 = vid_io.read_files(fns0)
    fns1 = result_p[1]['deno_fns'][0]
    vid1 = vid_io.read_files(fns1)

    # -- merge & save --
    regions,fps = region_clip(cfg_p[0],clean,noisy,vid0,vid1)
    # print(len(regions),"regions")
    for r_index,region in enumerate(regions):

        # -- optional skip --

        # -- crops dir --
        clean,noisy,vid0,vid1 = region
        vid = vid_merge(noisy,vid1,vid0,clean,nrow=4)
        crop_dir = gif_dir / ("crop_%d" % r_index)
        if not crop_dir.exists():
            crop_dir.mkdir()
        else:
            if not(overwrite): continue
        # print(crop_dir)
        vid_i = [th.from_numpy(np.array(v)) for v in vid]
        vid_i = rearrange(th.stack(vid_i),'t h w c -> t c h w')
        vid_io.save_video(vid_i,crop_dir,"crop")

# def create_lidia_gifs(base,version,exps,ow_gifs):
#     cache_dir = ".cache_io/test_lidia_noT_r2"
#     cache_name = "v1"
#     cache0 = cache_io.ExpCache(cache_dir,cache_name)
#     cache_dir = ".cache_io/test_lidia_yesT"
#     cache_name = "v1"
#     cache1 = cache_io.ExpCache(cache_dir,cache_name)
#     exps = dev_exps.load("exps/test_lidia_v2.cfg")#get_exps()
#     exps = [e for e in exps if e.sigma == 30]
#     exps = [e for e in exps if e.dname == "set8"]
#     # vnames = ["park_joy","sunflower"]
#     vnames = ["tractor"]
#     exps = [e for e in exps if e.vid_name in vnames]

#     u0,c0,r0 = cache0.load_raw_configs(exps,True)
#     u1,c1,r1 = cache1.load_raw_configs(exps,True)
#     u,c,r = make_lidia_pairs(u0,u1,c0,c1,r0,r1)
#     N = len(u)
#     for cfg_p,uuid_p,result_p in tqdm.tqdm(zip(c,u,r),total=N):
#         create_gif_res(base,version,cfg_p,uuid_p,result_p,ow_gifs)

# def make_lidia_pairs(u0,u1,c0,c1,r0,r1):
#     df = pd.DataFrame(c0 + c1)
#     fields = ["arch_name","vid_name","sigma"]
#     args0,args1 = [],[]
#     for f,gdf in df.groupby(fields):
#         if len(gdf) != 2: continue
#         for wt,wdf in gdf.groupby("wt"):
#             if wt == 0:
#                 args1.append(int(wdf.index[0]))
#             elif wt == 3:
#                 args0.append(int(wdf.index[0]))
#     print(len(args0))
#     print(len(args1))
#     uuids = split_args(u0+u1,args0,args1)
#     results = split_args(r0+r1,args0,args1)
#     configs = split_args(c0+c1,args0,args1)
#     return uuids,configs,results

def create_pairs(uuids,configs,results):

    # -- indices --
    df = pd.DataFrame(configs)
    args0,args1 = [],[]
    fields = ["arch_name","vid_name","sigma"]
    print(df.columns)
    for f,gdf in df.groupby(fields):
        for wt,wdf in gdf.groupby("wt"):
            # print(wdf[['vid_name','sigma']])
            if wt == 0:
                args1.append(int(wdf.index[0]))
            elif wt == 3:
                args0.append(int(wdf.index[0]))

    # -- pairs --
    uuids = split_args(uuids,args0,args1)
    results = split_args(results,args0,args1)
    configs = split_args(configs,args0,args1)
    # print(len(configs))

    return uuids,configs,results

def split_args(pylist,args0,args1):
    return [[pylist[i],pylist[j]] for i,j in zip(args0,args1)]


def replace_uuid(config_p,uuid_p,res_p):
    assert config_p[1].wt == 0
    uuid = uuid_p[1]
    cfg = config_p[1]
    res = res_p[1]
    if cfg.vid_name == "cows":
        uuid_r = "68cab5aa-8ad7-490c-aa79-eed4b673e20e"
        replace_fns(res,uuid,uuid_r)

def replace_fns(res,uuid0,uuid1):
    fns = res['deno_fns'][0]
    fns_r = []
    for fn in fns:
        fn_r = fn.replace(uuid0,uuid1)
        fns_r.append(fn_r)
    res['deno_fns'][0] = fns_r

def main():
    base = Path("output/organized_denos/")
    cache_dir = ".cache_io/test_colanet_01_02"
    # cache_dir = ".cache_io/test_nets_v9"
    cache_name = "v1"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # exps = dev_exps.load("exps/test_n3net.cfg")
    # exps = dev_exps.load("exps/test_lidia.cfg")
    exps = dev_exps.load("exps/test_colanet_old.cfg")
    # exps = [e for e in exps if e.sigma == 50]
    # exps = [e for e in exps if e.dname == "davis"]
    # vnames = ["dance-twirl","loading","parkour"]
    # vnames += ["goat","judo","libby","loading","breakdance","car-roundabout"]
    # vnames = ["breakdance"]
    vnames = ["judo"]
    vnames += ["sunflower"]
    vnames += ["dance-twirl"]
    vnames += ["tractor"]
    vnames += ["blackswan","bike-packing","soapbox"]
    vnames = ["drift-straight"]
    # vnames = ["bike-packing"]#,"soapbox"]
    # vnames = ["bike-packing"]
    # vnames = ["scooter-black","loading","camel"]
    # vnames = ["scooter-black"]
    # vnames = ["tractor"]
    sigma = [50]
    # vnames = ["cows"]
    # vnames = ["sunflower"]
    # exps = [e for e in exps if e.sigma in sigma]
    exps = [e for e in exps if e.vid_name in vnames]
    print(len(exps))
    # exps = [e for e in exps if e.wt == 3]
    version = 0
    ow_gifs = True
    # create_links(base,version,cache,exps)
    create_gifs(base,version,cache,exps,ow_gifs)
    create_crops(base,version,cache,exps,ow_gifs)
    # create_lidia_gifs(base,version,exps,ow_gifs)


if __name__ == "__main__":
    main()
