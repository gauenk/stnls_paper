
"""

Check the restored state with pytorch lighitning updates the learning rate.

"""


import torch
import torch as th
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning.profilers import SimpleProfiler

# -- wandb logger --
import uuid
import pandas as pd
import wandb


import os,shutil
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger,WandbLogger
from pytorch_lightning import Callback
# from lightning.pytorch.callbacks import Callback
# from pytorch_lightning.metrics.functional import accuracy
# tmpdir = os.getcwd()

from matplotlib import pyplot as plt

pl.seed_everything(42)

class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class ResumeCallback(Callback):

    def __init__(self,ckpt):
        self.ckpt = ckpt

    def on_train_start(self, trainer, pl_module):
        print("Resume!")
        if not(self.ckpt is None):
            self.resume_checkpoint(trainer,pl_module,self.ckpt)

    def resume_checkpoint(self,trainer,model,ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # print(list(checkpoint.keys()))
        # sched = trainer.lr_scheduler_configs[0].scheduler

        sched = model.lr_schedulers()
        sched.load_state_dict(checkpoint['lr_schedulers'][0])
        model.load_state_dict(checkpoint['state_dict'])

        # print(trainer.optimizers)
        # print(trainer.lr_scheduler_configs[0])
        # print(type(checkpoint['optimizer_states'][0]))
        # print(checkpoint['optimizer_states'][0])

        # print(trainer.optimizers[0])
        trainer.optimizers[0].load_state_dict(checkpoint['optimizer_states'][0])
        print(trainer.optimizers[0])

class BoringModel(LightningModule):

    def __init__(self,scheduler_name="cosine_annealing",ckpt_path=None):
        super().__init__()
        self.scheduler_name = scheduler_name
        self.layer = torch.nn.Linear(32, 2)
        self.ckpt_path = ckpt_path
        self.limit_train_batches = 600
        # self.automatic_optimization = False

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))

    def training_step(self, batch, batch_idx):

        # -- forward --
        output = self.layer(batch)
        loss = self.loss(batch, output)
        lr = self.optimizers()._optimizer.param_groups[0]['lr']
        # lr = self.optimizers().param_groups[-1]['lr']

        # # -- backward --
        # self.manual_backward(loss)

        # # -- accumulate --
        # self.accumulate_grad_batches = 1
        # agb = self.accumulate_grad_batches
        # if (batch_idx + 1) % agb == 0:
        #     self.optimizers().step()
        #     self.optimizers().zero_grad()

        # # -- scheduler --
        # sch = self.lr_schedulers()

        # # step every `n` batches
        # if (batch_idx + 1) % n == 0:
        #     sch.step()

        # # step every `n` epochs
        if self.trainer.is_last_batch:
            print("lr: ",lr)
            # sch.step()

        # -- log --
        # print(self.optimizers().state_dict())
        self.log("loss", loss)
        self.log("lr", lr)
        # print("lr: ",lr)
        return {"loss": loss,"lr": lr}

    # def on_train_start(self):
    #     print(dir(self.optimizers()))
    #     self.optimizers().state_dict = self.optimizers()._optimizer.state_dict
    #     self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def training_step_end(self, training_step_outputs):
        # lr = self.optimizers()._optimizer.param_groups[-1]['lr']
        # print(lr)
        return training_step_outputs

    # def training_epoch_end(self, outputs) -> None:
    #     torch.stack([x["loss"] for x in outputs]).mean()

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        if self.limit_train_batches > 0:
            dataset_size = self.limit_train_batches
            num_devices = 1
        else:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            num_devices = max(1, self.trainer.num_devices)
        acc = self.trainer.accumulate_grad_batches
        num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_steps

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    # def validation_epoch_end(self, outputs) -> None:
    #     torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    # def test_epoch_end(self, outputs) -> None:
    #     torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        # print("\n"*5)
        # print("HI!")
        # print("\n"*5)
        optimizer = torch.optim.Adam(self.layer.parameters(), lr=0.1)
        ckpt_path = self.ckpt_path
        if not(self.ckpt_path is None):
            checkpoint = torch.load(ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        lrs = torch.optim.lr_scheduler
        if self.scheduler_name == "cosine_annealing":
            num_steps = self.num_steps()
            print("num_steps: ",num_steps)
            # lr_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=num_steps)
            lr_scheduler = lrs.CosineAnnealingLR(optimizer, num_steps)
            scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["coswr","cosw"]:
            lr_sched =th.optim.lr_scheduler
            CosineAnnealingWarmRestarts = lr_sched.CosineAnnealingWarmRestarts
            # print(self.coswr_T0,self.coswr_Tmult,self.coswr_eta_min)
            coswr_T0 = 10*(self.num_steps() // self.trainer.max_epochs)
            coswr_Tmult = 2
            coswr_eta_min = 1e-9
            scheduler = CosineAnnealingWarmRestarts(optimizer,coswr_T0,
                                                    T_mult=coswr_Tmult,
                                                    eta_min=coswr_eta_min)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name == "multistep":
            milestones = [1,2,3,4]
            gamma = 0.5
            lr_scheduler = lrs.MultiStepLR(optimizer,milestones=milestones,
                                           gamma=gamma)
            scheduler = {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name == "exp":
            lr_scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
            scheduler = {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
        else:
            raise ValueError(f"Unknown scheduler [{self.scheduler_name}]")

        if not(self.ckpt_path is None):
            checkpoint = torch.load(ckpt_path)
            scheduler['scheduler'].load_state_dict(checkpoint['lr_schedulers'][0])
        return [optimizer],[scheduler]

def test_x(train,val,tmpdir,nepochs=5,scheduler_name="cosine_annealing"):

    # init model
    model = BoringModel(scheduler_name)

    # init callbacks
    model_checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=nepochs, monitor="loss",
                                                    dirpath=tmpdir/"checkpoints_x",
                                                    filename="dev-{epoch:02d}")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    callbacks = [model_checkpoint, lr_monitor]
    # callbacks = [lr_monitor]
    # logger = CSVLogger(tmpdir/"csv_logger_x", name="loggin")
    uuid_str = str(uuid.uuid4())
    name = "test_x_%s"%uuid_str
    print("name: ",name)
    RANK = int(os.environ.get('LOCAL_RANK', 0))
    if RANK == 0:
        logger = WandbLogger(name=name,project="testing_anvil")
    else:
        logger = None

    # -- profiling --
    prof = SimpleProfiler("train")

    # Initialize a trainer
    trainer = pl.Trainer(
        log_every_n_steps=1,
        limit_train_batches=600,
        max_epochs=nepochs,
        callbacks=callbacks,
        logger=logger,
        devices=4,
        # profiler=prof,
        # enable_checkpointing=False,
    )

    # Train the model ⚡
    trainer.fit(model, train, val)

    # Load Learning Rate
    api = wandb.Api()
    runs = api.runs("gauenk/testing_anvil")
    print("len(runs) :",len(runs))
    runs = [r for r in runs if r.name == name]
    print("len(runs) :",len(runs))
    print(dir(runs[0]))
    run = [r for r in runs if r.name == name][0]
    run.wait_until_finished()
    # print(dir(run))
    hist = run.history(samples=600*nepochs).sort_values("_runtime")
    print(hist)
    print(len(hist))
    lr = hist['lr'].to_numpy()
    print("[test_x] lr: ",lr.shape)

    # Load Learning Rate
    # lr = pd.read_csv(tmpdir/"csv_logger_x/loggin/version_0/metrics.csv")['lr']
    # lr = lr.dropna()
    # lr = lr.to_numpy()

    return lr

def test_y(train,val,tmpdir,nepochs=5,scheduler_name="cosine_annealing",
           resume_epoch=2,resume_name="x",name="y0"):

    # init model
    ckpt_path = tmpdir/f"checkpoints_{resume_name}/dev-epoch={resume_epoch:02d}.ckpt"
    model = BoringModel(scheduler_name)

    # init callbacks
    resume = ResumeCallback(ckpt_path)
    model_checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=nepochs, monitor="loss",
                                                    dirpath=tmpdir/f"checkpoints_{name}",
                                                    filename="dev-{epoch:02d}")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # callbacks = [resume,model_checkpoint,lr_monitor]
    callbacks = [model_checkpoint,lr_monitor]
    logger = WandbLogger(name="test_y",project="testing")
    # logger = CSVLogger(tmpdir/f"csv_logger_{name}", name="loggin")

    # Initialize a trainer
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=nepochs,
        callbacks=callbacks,
        logger=logger,
        devices=1,
    )

    # Train the model ⚡
    trainer.fit(model, train, val, ckpt_path=ckpt_path)

    # Load Learning Rate
    api = wandb.Api()
    runs = api.runs("gauenk/testing")
    run = [r for r in runs if r.name == "test_y"][0]
    hist = run.history()
    lr = hist['lr'].to_numpy()
    print("lr: ",lr.shape)

    # # Load Learning Rate
    # lr = pd.read_csv(tmpdir/f"csv_logger_{name}/loggin/version_0/metrics.csv")['lr']
    # lr = lr.dropna()
    # lr = lr.to_numpy()

    return lr

def test_z(train,val,tmpdir,nepochs=5,scheduler_name="cosine_annealing",
           resume_epoch=2,resume_name="x",name="z"):

    # init model
    ckpt_path = tmpdir/f"checkpoints_{resume_name}/dev-epoch={resume_epoch:02d}.ckpt"
    model = BoringModel(scheduler_name,ckpt_path=ckpt_path)

    # init callbacks
    resume = ResumeCallback(ckpt_path)
    model_checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=nepochs, monitor="loss",
                                                    dirpath=tmpdir/f"checkpoints_{name}",
                                                    filename="dev-{epoch:02d}")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # callbacks = [resume,model_checkpoint,lr_monitor]
    callbacks = [model_checkpoint,lr_monitor]
    logger = WandbLogger(name="test_z",project="testing")
    # logger = CSVLogger(tmpdir/f"csv_logger_{name}", name="loggin")

    # Initialize a trainer
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=nepochs,
        callbacks=callbacks,
        logger=logger,
        devices=1,
    )

    # Restore Checkpoint
    # checkpoint = torch.load(ckpt_path)
    # print(list(checkpoint.keys()))
    # print(dir(trainer))
    # model.load_state_dict(checkpoint['state_dict'])
    # print(list(trainer.lr_scheduler_configs))
    # print(trainer.optimizers,type(checkpoint['optimizer_states']))
    # trainer.optimizers.load_state_dict(checkpoint['optimizer_states'])
    # sched = trainer.lr_scheduler_configs['scheduler']
    # sched.load_state_dict(checkpoint['lr_schedulers'])
    # exit(0)
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # Train the model ⚡
    trainer.fit(model, train, val, ckpt_path=ckpt_path)

    # Load Learning Rate
    api = wandb.Api()
    runs = api.runs("gauenk/testing")
    run = [r for r in runs if r.name == "test_z"][0]
    hist = run.history()
    lr = hist['lr'].to_numpy()
    print("lr: ",lr.shape)

    # # Load Learning Rate
    # lr = pd.read_csv(tmpdir/f"csv_logger_{name}/loggin/version_0/metrics.csv")['lr']
    # lr = lr.dropna()
    # lr = lr.to_numpy()

    return lr

def load_weights(tmpdir,name,nepochs=5):
    ckpt_path = tmpdir/f"checkpoints_{name}/dev-epoch={(nepochs-1):02d}.ckpt"
    state = th.load(ckpt_path)['state_dict']
    # print(list(state.keys()))
    return state

def compare_weights(weights_a,weights_b):
    diff = 0.
    for key in weights_a:
        diff += th.mean((weights_a[key] - weights_b[key])**2).item()
    return diff

def viz_lrs(lr0,lr1,lr2):
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    nskip = len(lr0) - len(lr1)
    lr1 = np.r_[np.zeros(nskip),lr1]
    lr2 = np.r_[np.zeros(nskip),lr2]
    ax.plot(lr0,label="0")
    ax.plot(lr1,label="1")
    ax.plot(lr2,label="3")
    plt.savefig("output/dev/restore_state.png")
    plt.close("all")

def main():

    # -- init --
    print("PID: ",os.getpid())

    # -- datasets --
    num_samples = 2400
    train = RandomDataset(32, num_samples)
    train = DataLoader(train, batch_size=1, num_workers=4)
    val = RandomDataset(32, num_samples)
    val = DataLoader(val, batch_size=1)
    test = RandomDataset(32, num_samples)
    test = DataLoader(test, batch_size=1)

    # -- config --
    nepochs = 1000
    # scheduler_name = "multistep"
    scheduler_name = "cosine_annealing"
    # scheduler_name = "cosine_annealing"
    # scheduler_name = "cosw"

    # -- testing --
    RANK = int(os.environ.get('LOCAL_RANK', 0))
    output = Path("output/dev/lightning_logs/")
    if RANK == 0:
        if output.exists(): shutil.rmtree(str(output))
        output.mkdir(parents=True)
    lr0 = test_x(train,val,output,nepochs,scheduler_name)
    print(lr0)
    print(lr0.reshape(-1,num_samples).mean(-1))
    lr1 = test_y(train,val,output,nepochs,scheduler_name,
                   resume_epoch=int(0.5*nepochs),resume_name="x",name="y")
    lr1_1 = test_y(train,val,output,nepochs,scheduler_name,
                   resume_epoch=int(0.8*nepochs),resume_name="y",name="y1")
    lr2 = test_z(train,val,output,nepochs,scheduler_name,
                   resume_epoch=int(0.5*nepochs),resume_name="x",name="z")
    lr2_1 = test_z(train,val,output,nepochs,scheduler_name,
                   resume_epoch=int(0.8*nepochs),resume_name="z",name="z1")

    print(lr0)

    # -- check weights --
    weights_x = load_weights(output,"x",nepochs)
    weights_y = load_weights(output,"y",nepochs)
    weights_z = load_weights(output,"z",nepochs)
    weights_y1 = load_weights(output,"y1",nepochs)
    weights_z1 = load_weights(output,"z1",nepochs)
    diff_xy = compare_weights(weights_x,weights_y)
    diff_xz = compare_weights(weights_x,weights_z)
    diff_yy = compare_weights(weights_y,weights_y1)
    diff_zz = compare_weights(weights_z,weights_z1)
    # print(diff_xy)
    # print(diff_xz)
    # print(diff_yy)
    # print(diff_zz)

    # -- viz --
    print(lr0)
    print(lr1)
    print(lr2)
    viz_lrs(lr0,lr1,lr2)

    # -- check --
    start = len(lr0) - len(lr2)
    diff = np.mean(np.abs(lr0[start:] - lr2)).item()
    print(diff)
    assert diff < 1e-15

if __name__ == "__main__":
    main()
