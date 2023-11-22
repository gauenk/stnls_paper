"""

Check the quaity of a step-based trainer for seamless
resume on SLURM

"""

# -- basic --
import torch as th
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- pytorch-lit --
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_trainer(cfg,chkpt_dir):
    """
    Save based on global_step, not epoch.
    """
    logger = CSVLogger(cfg.log_dir,name="trainer",flush_logs_every_n_steps=1)
    ckpt_fn_val = cfg.uuid + "-{epoch:02d}-{val_loss:2.2e}"
    checkpoint_list = SaveCheckpointList(cfg.uuid,chkpt_dir,cfgs.tr.save_epoch_list)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=3,
                                          mode="min",dirpath=chkpt_dir,
                                          filename=ckpt_fn_val)
    ckpt_fn_epoch = cfgs.tr.uuid + "-{epoch:02d}"
    cc_recent = ModelCheckpoint(monitor="epoch",save_top_k=3,mode="max",
                                dirpath=chkpt_dir,filename=ckpt_fn_epoch)
    callbacks = [checkpoint_list,checkpoint_callback,cc_recent]
    trainer = pl.Trainer(accelerator="gpu",num_nodes=1,
                         devices=1,precision=32,
                         accumulate_grad_batches=cfg.accumulate_grad_batches,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=1.,
                         # max_epochs=cfgs.tr.nepochs,
                         max_steps=cfg.nsteps,
                         log_every_n_steps=1,
                         logger=logger,
                         gradient_clip_val=cfgs.tr.gradient_clip_val,
                         gradient_clip_algorithm=cfgs.tr.gradient_clip_algorithm,
                         callbacks=callbacks,
                         strategy="ddp_find_unused_parameters_false")

    return trainer,chkpt_callback

def get_cfg():

    cfg = edict()
    cfg.uuid = "example_uuid"
    cfg.accumulate_grad_batches = 1
    cfg.limit_train_batches = 1
    cfg.log_dir = "output/dev/step_based_trainer/log"
    cfg.root = "output/dev/step_based_trainer/"
    cfg.nsteps = 20
    cfg.nepochs = 10

def get_checkpoint(checkpoint_dir,uuid,nepochs):
    """
    Picks based on epochs, not steps
    """
    checkpoint_dir = Path(checkpoint_dir)
    if rank_zero_only.rank > 0:
        wait_checkpoint_exists(checkpoint_dir)
    else:
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
            return None
    chosen_ckpt = ""
    for epoch in range(nepochs):
        # if epoch > 49: break
        ckpt_fn = checkpoint_dir / ("%s-epoch=%02d.ckpt" % (uuid,epoch))
        if ckpt_fn.exists(): chosen_ckpt = ckpt_fn
    assert ((chosen_ckpt == "") or chosen_ckpt.exists())
    if chosen_ckpt != "":
        print("Resuming training from {%s}" % (str(chosen_ckpt)))
        chosen_ckpt = str(chosen_ckpt)
    else:
        chosen_ckpt = None
    return chosen_ckpt

def main():

    cfg = get_cfg()
    chkpt_dir = Path(cfg.root) / "checkpoints" / str(cfg.uuid)
    ckpt_path = get_checkpoint(chkpt_dir,cfg.uuid,cfg.nepochs)
    trainer,chkpt_callback = get_trainer(*args,**kwargs)
    print(len(loaders[dset_val]),type(loaders[dset_val]))
    print("Checkpoint Path: %s" % str(ckpt_path))
    trainer.fit(model, loaders[dset_tr], loaders[dset_val], ckpt_path=ckpt_path)
    best_model_path = chkpt_callback.best_model_path


if __name__ == "__main__":
    main()
