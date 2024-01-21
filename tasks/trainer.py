import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
from utils.metric import compute_errors


class MonoDepthTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(MonoDepthTrainer, self).__init__()
        self.model = model
        self.args = args
        self.save_hyperparameters(ignore=["model"])
        self.validation_outputs = list()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch, self.args.frame_ids)
        self.log("loss/train",
                 loss,
                 prog_bar=True,
                 on_epoch=True,
                 on_step=True,
                 batch_size=self.args.batch_size)
        del batch
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            images = batch[('raw', 0, 0)]
            disparity = self.model.predict_disparity(images)
            for i in range(min(4, self.args.batch_size)):
                self.logger.experiment.add_image("a_img_raw/img_{:d}".format(i), images[i], self.current_epoch)
                self.logger.experiment.add_image("a_img_dis/img_{:d}".format(i), disparity[i], self.current_epoch)

        if self.args.with_gt:
            inp = batch[('raw', 0, 0)].to(self.device)
            scaled_disp, depth = self.model.predict_depth(inp)
            errors = self.compute_eigen_metric(depth.squeeze(1).cpu().numpy(), batch['gt_data'].cpu().numpy())
            self.validation_outputs.append(torch.from_numpy(errors).to(self.device))
            loss = errors[0]
        else:
            loss = self.model(batch, self.args.frame_ids)
            self.validation_outputs.append(loss)
        self.log("loss_val", loss, batch_size=self.args.batch_size)
        del batch

    def on_validation_end(self):
        mean_loss = torch.stack(self.validation_outputs, dim=-1).mean(dim=-1)
        tensorboard = self.logger.experiment
        if mean_loss.numel() == 1:
            loss = mean_loss
            tensorboard.add_scalar("val/loss", loss, self.current_epoch)
        else:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_loss
            tensorboard.add_scalars("val", {"abs_rel": abs_rel,
                                            "sq_rel": sq_rel,
                                            "rmse": rmse,
                                            "rmse_log": rmse_log,
                                            "a1": a1,
                                            "a2": a2,
                                            "a3": a3
                                            }, self.current_epoch)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop, 0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler}
        }

    def forward(self, x):
        return self.model.predict_depth(x)

    def compute_eigen_metric(self, preds, gts):
        errors = list()
        for pred_depth, gt_depth in zip(preds, gts):
            gt_height, gt_width = gt_depth.shape
            mask = np.logical_and(gt_depth > self.args.min_thresh, gt_depth < self.args.max_thresh)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if self.args.strategy == "M":
                ratio = np.median(gt_depth) / np.median(pred_depth)
            else:
                ratio = 5.4
            pred_depth *= ratio
            pred_depth[pred_depth < self.args.min_thresh] = self.args.min_thresh
            pred_depth[pred_depth > self.args.max_thresh] = self.args.max_thresh
            errors.append(compute_errors(gt_depth, pred_depth))
        mean_errors = np.array(errors).mean(0)
        return mean_errors
