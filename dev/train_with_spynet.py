"""

  Verify we can train SpyNet after K iterations

"""

# -- sys --
import os,shutil
import torch as th
import numpy as np
import pandas as pd
import importlib
from pathlib import Path

# -- testing --
from dev_basics.trte import train,bench

# -- caching results --
import cache_io

def get_exp():
    exps,uuids = cache_io.train_stages.run("exps/trte_nlnet/train.cfg",
                                           ".cache_io_exps/trte_nlnet/train/",update=True)
    exp = exps[0]
    exp.chkpt_root = "output/train/dev_train_with_spynet"
    exp.name = "dev_train_with_spynet"
    exp.batch_size = 1
    exp.batch_size_tr = 1
    exp.accumulate_grad_batches = 1
    exp.limit_train_batches = 10
    exp.spynet_global_step = 25
    exp.nepochs = 5
    exp.flow = False
    exp.read_flows = False
    exp.save_epoch_list = "1-2-3-4-5"
    exp.use_wandb = False
    exp.nsamples_val = 1
    exp.uuid = "dev"
    return exp

def get_models(exp,chkpt_root):
    models = {}
    net_module = importlib.import_module(exp.python_module)
    for ckpt_path in sorted(chkpt_root.iterdir()):
        if not("save" in str(ckpt_path)): continue
        epoch = int(ckpt_path.name.split("=")[-1].split(".")[0])
        exp.pretrained_root = chkpt_root
        exp.pretrained_path = ckpt_path.name
        exp.pretrained_load = True
        models[str(epoch)] = net_module.load_model(exp)
    print(list(models.keys()))
    return models

def compare_paired_params(model_a,model_b):
    params_a = th.nn.utils.parameters_to_vector(model_a.parameters())
    params_b = th.nn.utils.parameters_to_vector(model_b.parameters())
    diff = th.mean((params_a - params_b)**2).item()
    return diff

def compare_pwd(models):
    diffs = np.zeros((len(models),len(models)))
    for i,key_i in enumerate(models):
        for j,key_j in enumerate(models):
            if j >= i: continue
            diffs[i,j] = compare_paired_params(models[key_i],models[key_j])
    return diffs

def get_spynets(models):
    spynets = {}
    for key in models:
        spynets[key] = models[key].spynet
    return spynets

def main():

    # -- config --
    output = Path("output/train/dev_train_with_spynet/")
    exp = get_exp()

    # -- run training --
    # if output.exists():
    #     shutil.rmtree(str(output))
    # res = train.run(exp)

    # -- compare parameters --
    ckpt_path = output / "checkpoints/dev/"
    models = get_models(exp,ckpt_path)
    spynets = get_spynets(models)
    model_diffs = compare_pwd(models)
    spynet_diffs = compare_pwd(spynets)
    # print(model_diffs)
    # print(spynet_diffs)

if __name__ == "__main__":
    main()
