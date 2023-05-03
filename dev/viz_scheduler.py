import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt

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
    lr = 5e-6
    optimizer = th.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    ChainedScheduler = th.optim.lr_scheduler.ChainedScheduler
    ExponentialLR = th.optim.lr_scheduler.ExponentialLR
    CosineAnnealingWarmRestarts = th.optim.lr_scheduler.CosineAnnealingWarmRestarts

    T_0 = 200
    T_mult = 2
    eta_min = 1e-8
    gamma = 0.99
    print(optimizer)

    # scheduler0 = ExponentialLR(optimizer, gamma=gamma)
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0,
                                             T_mult=T_mult,eta_min=eta_min)
    # scheduler = ChainedScheduler([scheduler1,scheduler0])
    scheduler = scheduler1
    print(scheduler)

    print(dir(scheduler))
    # print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
    #       scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
    #       scheduler.get_last_lr())
    print(scheduler.get_last_lr())
    scheduler.step()
    # print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
    #       scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
    #       scheduler.get_last_lr())
    print(scheduler.get_last_lr())
    nsteps = 1000
    lrs = collect_lrs(optimizer,scheduler,nsteps)
    print(len(lrs))
    plt.plot(lrs)
    root = Path("output/viz_schedulers/")
    if not(root.exists()): root.mkdir()
    plt.savefig(root / "lrs.png")

if __name__ == "__main__":
    main()
