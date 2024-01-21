import argparse


def get_train_parser():
    parser = argparse.ArgumentParser('Set Monodepthv2', add_help=False)
    parser.add_argument('--model_name', default="M_640x192", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=15, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--with_gt', default=True, type=bool)

    parser.add_argument('--kitti_dir', default="/home/lion/large_data/data/kitti/raw")
    parser.add_argument('--split', default="eigen_zhou", choices=["eigen_zhou", "eigen_full"])
    parser.add_argument('--frame_ids', nargs="+", default=[-1, 0, 1])
    parser.add_argument('--strategy', default='M', choices=['M', 'S', 'MS'])

    parser.add_argument('--scales', nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=192, type=int)
    parser.add_argument('--min_depth', default=0.1, type=float)
    parser.add_argument('--max_depth', default=100.0, type=float)
    parser.add_argument('--min_thresh', default=0.001, type=float)
    parser.add_argument('--max_thresh', default=80.0, type=float)

    return parser


# 0.109  &   0.836  &   4.814  &   0.198  &   0.868  &   0.956  &   0.980
def get_evaluation_parser():
    parser = argparse.ArgumentParser('Set Monodepthv2', add_help=False)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split_path', default="splits/eigen/test_files.txt", type=str)
    parser.add_argument('--gt_path', default="/home/lion/temp/gt_depths.npz", type=str)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser
