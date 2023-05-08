import math
import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

class NullModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1)

def collect_lrs(optim,scheduler,nsteps):
    lrs = []
    for i in range(nsteps):
        lr = optim.param_groups[0]['lr']
        scheduler.step()
        # print(scheduler.get_last_lr())
        lrs.append(lr)
    return np.array(lrs)

def main():

    model = NullModule()
    lr = 3e-4
    optimizer = th.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    # ChainedScheduler = th.optim.lr_scheduler.ChainedScheduler
    StepLR = th.optim.lr_scheduler.StepLR
    MultiStepLR = th.optim.lr_scheduler.MultiStepLR
    SequentialLR = th.optim.lr_scheduler.SequentialLR
    ExponentialLR = th.optim.lr_scheduler.ExponentialLR
    CosineAnnealingLR = th.optim.lr_scheduler.CosineAnnealingLR
    CosineAnnealingWarmRestarts = th.optim.lr_scheduler.CosineAnnealingWarmRestarts

    T_0 = 200
    T_mult = 2
    lr_final = 1e-6
    eta_min = 1e-8
    gamma = 0.9
    nsteps = 100
    print(optimizer)
    gamma = math.exp(math.log(lr_final/lr)/nsteps)
    print(gamma)

    # scheduler0 = ExponentialLR(optimizer, gamma=gamma)
    # scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0,
    #                                          T_mult=T_mult,eta_min=eta_min)
    # scheduler = SequentialLR(optimizer,[scheduler0,scheduler1],milestones=[100])
    # scheduler = ChainedScheduler([scheduler1,scheduler0],warmup_steps=10)
    # scheduler = scheduler1

    # scheduler = ChainedScheduler(
    #     optimizer,
    #     T_0 = T_0,
    #     T_mul = T_mult,
    #     eta_min = eta_min,
    #     gamma = gamma,
    #     max_lr = 1.0,
    #     warmup_steps= 100)
    # scheduler = CosineAnnealingLR(optimizer,100)
    # scheduler = ExponentialLR(optimizer, gamma=gamma)
    step_size = 5
    gamma = math.exp(math.log(lr_final/lr)/(nsteps//step_size))
    # scheduler = StepLR(optimizer,step_size=step_size,gamma=gamma)
    milestones = 10*np.arange(10)[2::2]
    print(milestones)
    scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=.5)
    # gamma = math.exp(math.log(lr_final/lr)/nsteps)
    # print(gamma)
    # scheduler = ExponentialLR(optimizer,gamma=gamma)

    print(scheduler)

    print(dir(scheduler))
    # print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
    #       scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
    #       scheduler.get_lr())
    # print(scheduler.get_lr())
    scheduler.step()
    # print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
    #       scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
    #       scheduler.get_lr())
    print(scheduler.get_lr())
    lrs = collect_lrs(optimizer,scheduler,nsteps)
    print(len(lrs))
    plt.plot(lrs)
    print(lrs[-1])
    root = Path("output/viz_schedulers/")
    if not(root.exists()): root.mkdir()
    plt.savefig(root / "lrs.png")

if __name__ == "__main__":
    main()
