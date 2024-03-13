import math
import argparse, yaml
# from . import sr_utils as utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict
from pathlib import Path
# from ..spa_config import config_via_spa
# from superpixel_paper.utils import hooks
from torchvision.utils import save_image,make_grid
import torch as th
# th.autograd.set_detect_anomaly(True)

from dev_basics import net_chunks
from easydict import EasyDict as edict
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR

import stnls
import data_hub
from einops import rearrange
import numpy as np

from .align_model import AlignModel
from .align_loss import AlignLoss
from .shared import seed_everything,optional
from . import utils

def extract_defaults(_cfg):
    cfg = edict(dcopy(_cfg))
    defs = {
        # "dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
        # "heads":1,"M":0.,"use_local":False,"use_inter":False,
        # "use_intra":True,"use_ffn":False,"use_nat":False,"nat_ksize":9,
        # "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        #
        "ws":10,"ps":1,"loss_fxn_input":"noisy",
        "align_type":"gda",
        # "align_type":"stnls",
        "nepochs":10,"attn_size":1,"ps_stack":3,
        "spynet_path":"./weights/spynet/spynet_sintel_final-3d2a1287.pth",
        "data_path":"./data/sr/","data_augment":False,
        "patch_size":512,"data_repeat":1,"threads":4,
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp","epochs":50,
        "log_every":10,"test_every":1,"batch_size":8,
        "log_path":"output/deno/train/","resume_uuid":None,"resume_flag":False}
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def load_data(_cfg):
    cfg = edict()
    cfg.seed = 123
    cfg.dname = "davis"
    cfg.nframes = 5
    cfg.sigma = _cfg.sigma
    cfg.isize = _cfg.isize
    # cfg.sigma = 0.001
    data,loaders = data_hub.sets.load(cfg)
    cfg.rand_order_tr = False
    _data,_loaders = data_hub.sets.load(cfg)
    return loaders.tr,_data.tr,_loaders.tr

def get_noisy(clean):
    # sigma_min = 5
    # sigma_max = 20
    # sigma = sigma_max*torch.rand(1).item()+sigma_min
    sigma = 15.
    # print(sigma)
    return clean + (sigma/255.) * th.randn_like(clean)

def run(cfg):

    # -- fill missing with defaults --
    cfg = extract_defaults(cfg)
    resume_uuid = cfg.uuid if cfg.resume_uuid is None else cfg.resume_uuid
    if cfg.resume_flag: cfg.resume = Path(cfg.log_path) / "checkpoints" / resume_uuid
    else: cfg.resume = None
    seed_everything(cfg.seed)
    device = torch.device('cuda')
    cfg.device = device
    torch.set_num_threads(cfg.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataset, valid_dataloaders = load_data(cfg)
    valid_dataloaders = [{"name":"davis","dataloader":valid_dataloaders}]
    tr_size = len(train_dataloader.dataset)

    ## definitions of model
    model = AlignModel(cfg,cfg.align_type,cfg.spynet_path)
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))
    print(model)
    # exit()
    model = nn.DataParallel(model).to(device)

    ## definition of loss and optimizer
    loss_func = AlignLoss(cfg)
    if len(list(model.parameters())) > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)
    else:
        optimizer = None
        scheduler = None

    # -- resume --
    stat_dict,start_epoch,chkpt_path,lpath = init_logging(cfg,model,optimizer,scheduler)

    ## start training
    timer_start = time.time()
    nval = 5

    # -- init validation --
    run_validation(cfg,model,stat_dict,-1,chkpt_path,lpath,
                   optimizer,scheduler,valid_dataset,valid_dataloaders,nval)

    for epoch in range(start_epoch, cfg.nepochs+1):

        # -- init epoch --
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = -1 if scheduler is None else scheduler.get_last_lr()
        th.manual_seed(int(cfg.seed)+epoch)
        print('##==========={}-training, Epoch: {}, lr: {} =============##'\
              .format('fp32', epoch, opt_lr))

        # -- run train --
        if not(optimizer is None):
            for iter, batch in enumerate(train_dataloader):
                vid = batch['clean'].to(device)/255.
                timer_start = run_batch(cfg,epoch_loss,vid,model,loss_func,
                                        optimizer,tr_size,
                                        timer_start,iter,epoch,stat_dict)
                # if iter > 30: break

        # -- run validation --
        if epoch % cfg.test_every == 0:
            run_validation(cfg,model,stat_dict,epoch,chkpt_path,lpath,
                           optimizer,scheduler,valid_dataset,valid_dataloaders,nval)

        # -- update scheduler --
        if not(scheduler is None):
            scheduler.step()

    # -- return info --
    info = edict()
    for valid_dataloader in valid_dataloaders:
        name = valid_dataloader['name']
        info["%s_best_psnr"%name] = stat_dict[name]['best_psnr']['value']
        info["%s_best_ssim"%name] = stat_dict[name]['best_ssim']['value']
    print(info)
    exit()
    return info

def run_batch(cfg,epoch_loss,vid,model,loss_func,optimizer,
              total_steps,timer_start,iter,epoch,stat_dict):
    optimizer.zero_grad()
    noisy = get_noisy(vid)
    flow_k = model(noisy)
    # print("flow_k.shape: ",flow_k.shape)
    loss = loss_func(vid,noisy,flow_k)
    # print(loss.item())
    loss.backward()
    optimizer.step()
    epoch_loss += float(loss)
    if (iter + 1) % cfg.log_every == 0:
        cur_steps = (iter + 1) * cfg.batch_size
        fill_width = math.ceil(math.log10(total_steps))
        cur_steps = str(cur_steps).zfill(fill_width)

        epoch_width = math.ceil(math.log10(cfg.nepochs))
        cur_epoch = str(epoch).zfill(epoch_width)

        avg_loss = epoch_loss / (iter + 1)
        stat_dict['losses'].append(avg_loss)

        timer_end = time.time()
        duration = timer_end - timer_start
        timer_start = timer_end
        print('Epoch:{}, {}/{}, loss: {:.8f}, time: {:.3f}'.\
              format(cur_epoch, cur_steps, total_steps, avg_loss, duration))
    return timer_start

def run_validation(cfg,model,stat_dict,epoch,chkpt_path,lpath,
                   optimizer,scheduler,dataset,valid_dataloaders,nval):
    avg_psnr, avg_ssim = 0.0, 0.0
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    test_log = ''
    model = model.eval()
    stacking = stnls.agg.NonLocalGather(cfg.ps_stack,cfg.stride0,itype="float")
    for valid_dataloader in valid_dataloaders:
        avg_psnr, avg_ssim = 0.0, 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        if nval == -1: nval = len(loader)-1
        # print("len(loader): ",len(loader))
        # seed_everything(123)
        th.manual_seed(123)
        # batch_idx = 0
        # for batch in tqdm(loader, ncols=80):
        loader_it = iter(loader)
        for batch_idx in tqdm(range(nval), ncols=80):
            batch = next(loader_it)
            # print(dataset.groups[dataset.indices[batch['index'][0].item()]])
            noisy = batch['noisy'].to(cfg.device)/255.
            vid = batch['clean'].to(cfg.device)/255.
            # print(th.mean((noisy-vid)**2).sqrt()*255.)
            # print(vid[0,:,0,:2,:2])

            # print("vid.shape: ",vid.shape)
            torch.cuda.empty_cache()
            with th.no_grad():
                # noisy = get_noisy(vid)
                flows = model(noisy)
            # print("flows.shape: ",flows.shape)
            # print(flows[0,0,1])

            # -- warping --
            # flows = rearrange(flows,'b hd k t tr h w -> b hd t h w k tr')
            flows = flows.contiguous()
            flows[...,1:] = flows[...,1:].flip(-1)
            # print("vid.shape,flows.shape: ",vid.shape,flows.shape)
            ones = th.ones_like(flows[...,0])
            stack = stacking(vid,ones,flows)#[:,0]

            # -- compute metrics --
            # print("vid.shape,stack.shape: ",vid.shape,stack.shape)
            psnr = utils.calc_psnr(vid,stack)
            ssim = utils.calc_ssim(vid,stack)
            # print(psnr)
            # print(ssim)
            # exit()
            avg_psnr += np.mean(psnr).item()
            avg_ssim += np.mean(ssim).item()

            if batch_idx > nval: break
            # batch_idx += 1

        avg_psnr = round(avg_psnr/nval + 5e-3, 2)
        avg_ssim = round(avg_ssim/nval + 5e-5, 4)
        # print(stat_dict[name])
        stat_dict[name]['psnrs'].append(avg_psnr)
        stat_dict[name]['ssims'].append(avg_ssim)

        if stat_dict[name]['best_ssim']['value'] > 0.98:
            stat_dict[name]['best_ssim']['value'] = avg_ssim
            stat_dict[name]['best_ssim']['epoch'] = epoch
        if stat_dict[name]['best_psnr']['value'] < avg_psnr:
            stat_dict[name]['best_psnr']['value'] = avg_psnr
            stat_dict[name]['best_psnr']['epoch'] = epoch
        if stat_dict[name]['best_ssim']['value'] < avg_ssim:
            stat_dict[name]['best_ssim']['value'] = avg_ssim
            stat_dict[name]['best_ssim']['epoch'] = epoch
        args = (name,float(avg_psnr), float(avg_ssim),
                stat_dict[name]['best_psnr']['value'],
                stat_dict[name]['best_ssim']['value'],
                stat_dict[name]['best_psnr']['epoch'],
                stat_dict[name]['best_ssim']['epoch'])
        test_log += '[{}], PSNR/SSIM: {:.2f}/{:.4f} \
        (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(*args)
        print(test_log)
        sys.stdout.flush()

    # -- save model --
    model_str = '%s-epoch=%02d.ckpt'%(cfg.uuid,epoch-1) # "start at 0"
    saved_model_path = os.path.join(chkpt_path,model_str)
    print("saveing model at path [%s]" % str(saved_model_path))
    if not(optimizer is None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'stat_dict': stat_dict
        }, saved_model_path)
        torch.set_grad_enabled(True)

        # -- save state dict --
        logging_path = lpath
        stat_dict_name = os.path.join(logging_path, 'stat_dict_%d.yml' % epoch)
        with open(stat_dict_name, 'w') as stat_dict_file:
            yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)


def init_logging(cfg,model,optimizer,scheduler):

    ## resume training
    start_epoch = 1
    if cfg.resume is not None:
        # print(cfg.resume)
        chkpt_files = glob.glob(os.path.join(cfg.resume, "*.ckpt"))
        # print(chkpt_files)
        if len(chkpt_files) != 0:
            chkpt_files = sorted(chkpt_files, key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
            print("Resuming from ",chkpt_files[-1])
            chkpt = torch.load(chkpt_files[-1])
            prev_epoch = chkpt['epoch']
            start_epoch = prev_epoch + 1
            model.load_state_dict(chkpt['model_state_dict'])
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            scheduler.load_state_dict(chkpt['scheduler_state_dict'])
            stat_dict = chkpt['stat_dict']
            ## reset folder and param
            # experiment_path = cfg.resume
            experiment_name = cfg.uuid
            logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
            chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
            log_name = os.path.join(logging_path,'log.txt')
            print('select {}, resume training from epoch {}.'\
                  .format(chkpt_files[-1], start_epoch))
            # if not os.path.exists(logging_path): os.makedirs(logging_path)
            # if not os.path.exists(chkpt_path): os.makedirs(chkpt_path)
    else:
        ## auto-generate the output logname
        experiment_name = cfg.uuid
        timestamp = utils.cur_timestamp_str()
        # if cfg.log_name is None:
        #     experiment_name = '{}-{}-{}-x{}-{}'.format(cfg.exp_name, cfg.model, 'fp32', cfg.upscale, timestamp)
        # else:
        #     experiment_name = '{}-{}'.format(cfg.log_name, timestamp)
        # experiment_path = os.path.join(cfg.log_path, experiment_name)
        logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
        chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
        log_name = os.path.join(logging_path,'log.txt')
        stat_dict = utils.get_stat_dict(["davis"])
        ## create folder for chkpt and stat

    # -- init log --
    if not os.path.exists(logging_path): os.makedirs(logging_path)
    if not os.path.exists(chkpt_path): os.makedirs(chkpt_path)
    # experiment_model_path = os.path.join(experiment_path, 'checkpoints')
    # if not os.path.exists(experiment_model_path):
    #     os.makedirs(experiment_model_path)
    ## save training paramters
    exp_params = vars(cfg)
    exp_params_name = os.path.join(logging_path,'config.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)


    ## print architecture of model
    time.sleep(3) # sleep 3 seconds
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    # print(model)
    # exit()
    sys.stdout.flush()
    return stat_dict,start_epoch,chkpt_path,logging_path
