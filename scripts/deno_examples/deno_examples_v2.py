"""

Run models to get deno examples

"""

# -- misc --
import os,math,tqdm
import pprint,random,copy
pp = pprint.PrettyPrinter(indent=4)
from functools import partial

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- images --
from PIL import Image
from torchvision.utils import make_grid


# -- data --
import data_hub

# -- dev basics --
import dev_basics.exps as dev_exps
from dev_basics.reports import deno_report
from dev_basics.utils import vid_io
from dev_basics.utils.metrics import compute_psnrs,compute_ssims


# -- caching results --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import reports
from icml23 import deno_examples

def get_dir(base,version,cfg):
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

def read_nc_pair(cfg):
    data,loaders = data_hub.sets.load(cfg)
    index = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                   cfg.frame_start,cfg.frame_end)[0]
    clean = data[cfg.dset][index]['clean']
    noisy = data[cfg.dset][index]['noisy']
    return clean,noisy

def filter_exps(cfg,in_exps):
    fields = ["vid_name","sigma"]
    for field in fields:
        if len(in_exps) == 0: break
        in_exps = [e for e in in_exps if e[field] == cfg[field]]
    if len(in_exps) == 0:
        msg = "Filtered exps are zero so no deno examples saved."
        print(msg)
        exit(0)
    return in_exps

def read_metrics(row):
    m = edict()
    m.psnrs = np.mean(row['psnrs'].to_numpy()[0])
    m.ssims = np.mean(row['ssims'].to_numpy()[0])
    m.strred = np.mean(row['strred'].to_numpy()[0])
    m.wt = row['wt'].to_numpy()[0]
    m.sc = row['spatial_chunk_size'].to_numpy()[0]
    m.so = row['spatial_chunk_overlap'].to_numpy()[0]
    m.vid_name = row['vid_name'].to_numpy()[0]
    m.sigma = row['sigma'].to_numpy()[0]
    return m

def process_group(df):
    df = df.sort_values(by="groups")
    vids = []
    metrics = []
    for _,gdf in df.groupby("groups"):
        fns = gdf['deno_fns'].to_numpy()[0][0]
        metrics_r = read_metrics(gdf)
        vid_g = vid_io.read_files(fns)
        vids.append(vid_g)
        metrics.append(metrics_r)
    return vids,metrics

def vid_merge(*vids,nrow=6):
    vid = []
    for frames in zip(*vids):
        grid = make_grid(list(frames),nrow=nrow,pad_value=255)
        grid = grid.numpy().astype(np.uint8)
        grid = rearrange(grid,'c h w -> h w c')
        grid = Image.fromarray(grid)
        vid.append(grid)
    return vid

def save_deno_example(df,base,overwrite=True):

    # -- only have two --
    # print(df.head())
    assert len(df) == 4,f"Should only be four [{len(df)}]"

    # -- name groups --
    groups = np.zeros(len(df))
    g3 = df['wt'] == 3
    groups = np.where(g3,3,groups)
    g2 = np.logical_and(df['wt'] == 0,df['spatial_chunk_size'] == 0)
    groups = np.where(g2,2,groups)
    g1 = df['spatial_chunk_overlap'] > 0
    groups = np.where(g1,1,groups)
    df['groups'] = groups

    # -- read output --
    vids,metrics = process_group(df)

    # -- read noisy/clean --
    cfg = edict(df.iloc[0].to_dict())
    clean,noisy = read_nc_pair(cfg)
    vids.append(clean)
    vids.append(noisy)

    # -- append noisy --
    npsnr = np.mean(compute_psnrs(noisy,clean,255.))
    for m in metrics:
        m.npsnr = npsnr

    # -- collect regions --
    iregions,fps = region_clip(cfg)
    regions = cropped_at(iregions,*vids)
    out_dir = get_dir(base,0,cfg)
    for r_index,region in enumerate(regions):

        # -- crops dir --
        vid0,vid1,vid2,vid3,clean,noisy = region
        vid = vid_merge(noisy,vid0,vid1,vid2,vid3,clean,nrow=6)
        crop_dir = out_dir / ("crop_%d" % r_index)
        if not crop_dir.exists():
            crop_dir.mkdir(parents=True)
        else:
            if not(overwrite): continue
        vid_i = [th.from_numpy(np.array(v)) for v in vid]
        vid_i = rearrange(th.stack(vid_i),'t h w c -> t c h w')
        vid_io.save_video(vid_i,crop_dir,"crop")
    return metrics

def region_clip(cfg):
    fps = 1.
    regions = [None,[200,456,200,456]]
    if cfg.vid_name == "sunflower":
        regions += [[210,298,406,490]]
    elif cfg.vid_name == "tractor":
        # regions += [[180,336,100,256],[200,264,580,644]]
        regions += [[180,336,100,256],[200,264,600,664]]
    elif cfg.vid_name == "park_joy":
        regions += [[220,314,220,314]]
    elif cfg.vid_name == "bike-packing":
        regions += [[130,130+90,800,890]]
    elif cfg.vid_name == "judo":
        regions += [[0,200,300,500]]
    elif cfg.vid_name == "breakdance":
        regions += [[150,270,0,100],[175,295,100,200],[150,250,525,650]]
    elif cfg.vid_name == "scooter-black":
        fps = 0.3
        regions += [[50,50+64,400,464]]
    elif cfg.vid_name == "dance-twirl":
        fps = 0.3
        regions += [[50,50+128,325,325+128]]
    elif cfg.vid_name == "mbike-trick":
        fps = 0.3
        regions += [[150,150+128,420,420+128]]
    elif cfg.vid_name == "motocross-jump":
        fps = 0.3
        regions += [[164-32+32-8,100+128-32-8,564+16,500+128-16]]
    elif cfg.vid_name == "cows":
        fps = 1
        regions += [[100,356,400,656],[288,384,384,480]]
    return regions,fps

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

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io/deno_examples_colanet/"
    cache_name = "v1"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- run exps --
    # exps = dev_exps.load("exps/deno_examples_colanet.cfg")
    exps = cache_io.exps.load("exps/deno_examples_colanet.cfg")
    df= pd.DataFrame(exps)
    # fields = ['sigma','vid_name','wt','spatial_chunk_size','spatial_chunk_overlap','pretrained_path']
    # print(df[fields].to_csv("temp.csv",index=False))
    print(len(exps))

    # -- run --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):
        break

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        exp.uuid = uuid
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = test_model.run(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- save --
    df = cache.to_records(exps)
    # df = df[df['vid_name'].isin(["scooter-black"])]
    print(df[['vid_name','dname','sigma','deno_fns']])
    df = df[df['vid_name'].isin(["tractor","bike-packing","hypersmooth"])]
    # print(df['psnrs'])
    # print(len(df))
    fields = ["sigma","vid_name"]
    base = Path("output/organized_denos_v2")
    metrics = []
    for group,gdf in df.groupby(fields):
        sigma,vid_name = group
        base_g = base / vid_name / ("sigma_%d" % sigma)
        print(sigma,vid_name,base_g)
        metrics_g = save_deno_example(gdf,base_g)
        metrics.extend(metrics_g)
    metrics = pd.DataFrame(metrics)
    print(metrics)
    for group,gdf in metrics.groupby(fields):
        psnrs = list(gdf['psnrs'].to_numpy())
        print(group," & ".join([r"\hspace{.9cm} "+ "%2.2f" % p for p in psnrs]))


if __name__ == "__main__":
    main()
