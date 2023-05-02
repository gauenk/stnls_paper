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
        # print(optim)
        # lr = optim.param_groups[0]['lr']
        # scheduler.step(i)
        scheduler.step()
        print(scheduler.get_last_lr())
        # lrs.append(lr)
    return np.array(lrs)

def main():

    model = NullModule()
    lr = 5e-6
    optimizer = th.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    CosineAnnealingWarmRestarts = th.optim.lr_scheduler.CosineAnnealingWarmRestarts
    T_0 = 200
    T_mult = 2
    eta_min = 1e-8
    print(optimizer)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult,
                                            eta_min=eta_min)
                                            # last_epoch=-1, verbose=False)

    print(scheduler)

    print(dir(scheduler))
    print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
          scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
          scheduler.get_last_lr())
    scheduler.step(20)
    print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,
          scheduler.T_mult,scheduler._step_count,scheduler.eta_min,
          scheduler.get_last_lr())
    # print(scheduler.T_0,scheduler.T_cur,scheduler.T_i,scheduler.T_mult)
    return
    nsteps = 1000
    lrs = collect_lrs(optimizer,scheduler,nsteps)
    print(len(lrs))
    plt.plot(lrs)
    root = Path("output/viz_schedulers/")
    if not(root.exists()): root.mkdir()
    plt.savefig(root / "lrs.png")

if __name__ == "__main__":
    main()
