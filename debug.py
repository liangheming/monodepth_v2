import os
import argparse
from options import get_train_parser
from datasets.kitti import build_dataset
from torch.utils.data.dataloader import DataLoader
from models.monodepthv2 import build_model


def main(args):
    assert len(args.frame_ids) == 3 and 0 in args.frame_ids
    if args.strategy == "S":
        args.frame_ids = [0, 's']
    elif args.strategy == "MS":
        args.frame_ids.append('s')
    v_data = build_dataset(args, is_train=True)
    v_loader = DataLoader(dataset=v_data,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False, drop_last=False, collate_fn=v_data.collate_fn)
    model = build_model(args)
    for batch in v_loader:
        # print(batch.keys())
        loss = model(batch, args.frame_ids)
        print(loss)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MonoDepth Train", parents=[get_train_parser()])
    train_args = parser.parse_args()
    main(train_args)
