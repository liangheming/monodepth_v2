import os
import torch
import argparse
import pytorch_lightning as pl
from tasks.trainer import MonoDepthTrainer
from options import get_train_parser

from datasets.kitti import build_dataset
from models.monodepthv2 import build_model
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(1994)
# if support
torch.set_float32_matmul_precision('high')


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def main(args):
    assert len(args.frame_ids) == 3 and 0 in args.frame_ids
    if args.strategy == "S":
        args.frame_ids = [0, 's']
    elif args.strategy == "MS":
        args.frame_ids.append('s')
    t_data = build_dataset(args, is_train=True)
    v_data = build_dataset(args, is_train=False)
    args.with_gt = v_data.gt_data is not None
    t_loader = DataLoader(dataset=t_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          num_workers=args.num_workers,
                          collate_fn=t_data.collate_fn)
    v_loader = DataLoader(dataset=v_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=True,
                          num_workers=args.num_workers,
                          collate_fn=v_data.collate_fn)
    logger = TensorBoardLogger(
        save_dir="workspace",
        name=args.model_name,
        default_hp_metric=False
    )
    bar_callback = ProgressBar(refresh_rate=5)
    ckpt_dir = "workspace/{:s}/version_{:d}".format(args.model_name, logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{loss_val:.4f}',
                                          monitor='loss_val',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=8)
    model = build_model(args)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    wrapper = MonoDepthTrainer(model=model, args=args)
    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        accelerator='gpu',
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[bar_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        benchmark=True,
    )
    trainer.fit(wrapper, train_dataloaders=t_loader, val_dataloaders=v_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MonoDepth Train", parents=[get_train_parser()])
    train_args = parser.parse_args()
    main(train_args)
