import torch as th
import math
import argparse, yaml
import spin.utils as utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from dev_basics.utils.misc import set_seed

import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
# from spin.datas.utils import create_datasets
# from superpixel_paper.sr_datas.utils import create_datasets
from .shared import seed_everything,optional
from . import utils

def extract_defaults(_cfg):
    cfg = edict(dcopy(_cfg))
    defs = {
        "dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
        "heads":1,"M":0.,"use_local":False,"use_inter":False,
        "use_intra":True,"use_fnn":False,"use_nat":False,"nat_ksize":9,
        "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        "data_path":"./data/sr/","data_augment":False,
        "patch_size":128,"data_repeat":1,"eval_sets":["Set5"],
        "gpu_ids":"[1]","threads":4,"model":"model",
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp",
        "upscale":2,"epochs":50,"denoise":False,
        "log_every":100,"test_every":1,"batch_size":8,"sigma":25,"colors":3,
        "log_path":"output/deno/train/","resume_uuid":None,"resume_flag":True,
        "output_folder":"output/deno/test","save_output":False}
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def load_data(cfg):
    pass

def run(cfg):

    # -- setup --
    cfg = extract_defaults(cfg)
    seed_everything(cfg.seed)
    resume_uuid = cfg.tr_uuid if cfg.resume_uuid is None else cfg.resume_uuid
    if cfg.resume_flag: cfg.resume = Path(cfg.log_path) / "checkpoints" / resume_uuid
    else: cfg.resume = None
    test_epoch = int(cfg.pretrained_path.split("=")[-1].split(".")[0])
    out_base = "%d/%s/epoch=%02d" %(cfg.sigma,cfg.tr_uuid[:5],test_epoch)
    output_folder = Path(cfg.output_folder) / out_base
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    device = torch.device('cuda')
    torch.set_num_threads(cfg.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = load_data(cfg)
    tr_size = len(train_dataloader.dataset)

    ## definitions of model
    model = AlignModel(cfg.align_type)
    model = nn.DataParallel(model).to(device)
    read_checkpoint(cfg,model) # resume

    torch.set_grad_enabled(False)
    epoch = 1
    test_log = ''
    model = model.eval()
    info = {"dname":[],"name":[],"psnrs":[],"ssims":[]}
    for valid_dataloader in valid_dataloaders:
        avg_align = 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        data_i = valid_dataloader['data'] if "data" in valid_dataloader else None
        count = 0
        for vid in tqdm(loader, ncols=80):
            count += 1
            torch.cuda.empty_cache()
            with th.no_grad():
                flows = model(vid)

            output_name = os.path.join(str(output_folder), str(name))
            if not os.path.exists(output_name):
                os.makedirs(output_name)
            output_name = os.path.join(output_name,
                                         str(count) + '_x' + str(cfg.upscale) + '.png')

            psnr = utils.calc_psnr(vid,flows)
            ssim = utils.calc_ssim(vid,flows)
            avg_psnr += psnr
            avg_ssim += ssim

            # print("-"*20)
            if not(data_i is None):
                info['dname'].append(name)
                info['name'].append(Path(data_i.lr_filenames[count-1]).stem)
                info['psnrs'].append(psnr)
                info['ssims'].append(ssim)
                # print(data_i.lr_filenames[count-1],psnr,ssim)

        avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
        avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)

        if not(name in stat_dict):
            subkeys = ["psnrs","ssims"]
            stat_dict[name] = {}
            for k in subkeys: stat_dict[name][k] = []
            subkeys = ["best_psnr","best_ssim"]
            for k in subkeys: stat_dict[name][k] = {"value":0}

        stat_dict[name]['psnrs'].append(avg_psnr)
        stat_dict[name]['ssims'].append(avg_ssim)
        if stat_dict[name]['best_psnr']['value'] < avg_psnr:
            stat_dict[name]['best_psnr']['value'] = avg_psnr
            stat_dict[name]['best_psnr']['epoch'] = epoch
        if stat_dict[name]['best_ssim']['value'] < avg_ssim:
            stat_dict[name]['best_ssim']['value'] = avg_ssim
            stat_dict[name]['best_ssim']['epoch'] = epoch
        test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(name, cfg.upscale, float(avg_psnr), float(avg_ssim), 
            stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
            stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])
            stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])

    print(test_log)
    info = pd.DataFrame(info)
    for dname,df in info.groupby("dname"):
        print("----- %s -----"%dname)
        print(df)
    # print(opt,cfg)
    # # if cfg.save_to_tmp:
    # #     info.to_csv(".tmp/results.csv",index=False)
    # print(info.to_dict(orient="records"))
    # exit()
    info = info.rename(columns={"name":"iname"})

    return info.to_dict(orient="records")


def read_checkpoint(cfg,model):
    assert cfg.resume is not None
    chkpt_files = glob.glob(os.path.join(cfg.resume, "*.ckpt"))
    chkpt_fn = os.path.join(cfg.resume,cfg.pretrained_path)
    print("Checkpoint: ",chkpt_fn)
    ckpt = torch.load(chkpt_fn)
    prev_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])
    stat_dict = ckpt['stat_dict']
    print('select {} for testing'.format(chkpt_fn))
    time.sleep(3) # sleep 3 seconds
    return model

def cleanup_mstate(mstate):
    states = {}
    for key,state in mstate.items():
        if "cached_penc" in key: continue
        states[key] = state
    return states
