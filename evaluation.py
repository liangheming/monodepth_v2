import os
import torch
import argparse
import numpy as np
import torch.nn.functional as f
import torchvision.transforms.transforms as tf
from tqdm import tqdm
from datasets.kitti import pil_loader
from utils.metric import compute_errors
from models.monodepthv2 import build_model
from tasks.trainer import MonoDepthTrainer
from options import get_train_parser, get_evaluation_parser

STEREO_SCALE_FACTOR = 5.4


@torch.no_grad()
def main(args):
    print("{:s} Evaluation Start".format(args.model_name))
    print("ckpt: {:s}".format(args.ckpt))
    kitti_data_dir = args.kitti_dir
    split_txt = args.split_path
    img_paths = list()
    with open(split_txt, "r") as rf:
        split_lines = rf.readlines()
    for line in split_lines:
        folder, frame_id, _ = line.split()
        img_path = os.path.join(kitti_data_dir, folder, "image_02/data", "{:s}.jpg".format(frame_id))
        assert os.path.exists(img_path)
        img_paths.append(img_path)

    model = build_model(args)
    wrapper = MonoDepthTrainer(model=model, args=args)
    ckpt_path = args.ckpt
    wrapper.load_from_checkpoint(checkpoint_path=ckpt_path,
                                 map_location="cpu",
                                 model=model,
                                 args=args,
                                 strict=False)
    wrapper.eval()
    if args.cuda:
        wrapper.cuda()
    transform = tf.Compose([tf.Resize([args.height,
                                       args.width],
                                      interpolation=tf.InterpolationMode.NEAREST),
                            tf.ToTensor()])
    gt_data = np.load(args.gt_path, allow_pickle=True)["data"]
    errors = list()
    for gt_depth, img_path in tqdm(zip(gt_data, img_paths)):
        gt_height, gt_width = gt_depth.shape
        img_pil = pil_loader(img_path)
        inp = transform(img_pil)
        disp, _ = wrapper(inp.unsqueeze(0).cuda() if args.cuda else inp.unsqueeze(0))
        pred_disp = f.interpolate(disp, gt_depth.shape, mode="bilinear", align_corners=False)
        pred_depth = (1 / pred_disp).squeeze().cpu().numpy()
        mask = np.logical_and(gt_depth > args.min_thresh, gt_depth < args.max_thresh)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if args.strategy == "M":
            ratio = np.median(gt_depth) / np.median(pred_depth)
        else:
            ratio = STEREO_SCALE_FACTOR
        pred_depth *= ratio
        pred_depth[pred_depth < args.min_thresh] = args.min_thresh
        pred_depth[pred_depth > args.max_thresh] = args.max_thresh
        errors.append(compute_errors(gt_depth, pred_depth))
    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MonoDepth Evaluation", parents=[get_train_parser(), get_evaluation_parser()])
    evaluation_args = parser.parse_args()
    main(evaluation_args)
