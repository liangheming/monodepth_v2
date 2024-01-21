import torch
from torch import nn
from torch.nn import functional as f
from models.layers import PoseNet, DepthNet
from utils.geometry import transformation_parameters
from utils.loss import ProjectLoss, SmoothLoss


class Strategy:
    MONO_ONLY = 0
    STEREO_ONLY = 1
    BOTH = 2


class MonoDepth(nn.Module):
    def __init__(self,
                 scales=range(4),
                 min_depth=0.1,
                 max_depth=100.0,
                 input_height=192,
                 input_width=640,
                 strategy=Strategy.MONO_ONLY):
        super(MonoDepth, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.input_height = input_height
        self.input_width = input_width
        self.scales = scales
        self.depth = DepthNet(scales)
        self.strategy = strategy
        self.pose = None
        if strategy != Strategy.STEREO_ONLY:
            self.pose = PoseNet()
        mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False)[:, None, None].float()
        std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False)[:, None, None].float()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if strategy == Strategy.MONO_ONLY or strategy == Strategy.BOTH:
            pair_mean = torch.cat([mean, mean], dim=0)
            pair_std = torch.cat([std, std], dim=0)
            self.register_buffer("pair_mean", pair_mean)
            self.register_buffer("pair_std", pair_std)

        self.build_grid()
        self.project_loss = ProjectLoss()
        self.smooth_loss = SmoothLoss()

    def build_grid(self):
        y, x = torch.meshgrid(torch.arange(self.input_height), torch.arange(self.input_width), indexing="ij")
        mesh = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().requires_grad_(False)
        self.register_buffer("homo_mesh", mesh)

    def forward(self, nested_inp, frame_ids):
        raw_view = nested_inp[('raw', 0, 0)]
        depth_inp = nested_inp[('aug', 0, 0)]
        k = nested_inp['k']
        inv_k = nested_inp['inv_k']
        depth_inp = (depth_inp - self.mean) / self.std
        disparities = self.depth(depth_inp)
        resized_disparities = [
            f.interpolate(disp,
                          size=(self.input_height, self.input_width),
                          mode="bilinear",
                          align_corners=False)
            for disp in disparities
        ]
        depths = [self.disp_to_depth(disp)[-1] for disp in resized_disparities]
        synthetic_targets = self.get_synthetic(nested_inp, frame_ids)
        recon_loss = 0.0
        smooth_loss = 0.0
        for s in self.scales:
            identity_losses = list()
            projected_losses = list()
            depth, disp = depths[s], disparities[s]
            for view, r, t in synthetic_targets:
                synthetic_view = self.synthetic_from_view(depth, view, r, t, k, inv_k)
                projected_losses.append(self.project_loss(synthetic_view, raw_view))
                identity_losses.append(self.project_loss(view, raw_view))
            if len(synthetic_targets) > 1:
                projected_losses, _ = torch.min(torch.stack(projected_losses, dim=1), dim=1)
                identity_losses, _ = torch.min(torch.stack(identity_losses, dim=1), dim=1)
            else:
                projected_losses = projected_losses[0]
                identity_losses = identity_losses[0]
            noise = torch.randn(identity_losses.shape, device=identity_losses.device) * 0.00001
            mask = (projected_losses.detach() < (identity_losses + noise))
            valid_loss = projected_losses[mask].mean()
            smooth_loss += 0.001 * self.smooth_loss(disp, nested_inp['raw', 0, s]) / (2 ** s)
            recon_loss += valid_loss

        total_loss = (recon_loss + smooth_loss) / len(self.scales)
        return total_loss

    # def forward_raw(self, nested_inp, frame_ids):
    #     raw_view = nested_inp[('raw', 0, 0)]
    #     depth_inp = nested_inp[('aug', 0, 0)]
    #     k = nested_inp['k']
    #     inv_k = nested_inp['inv_k']
    #
    #     depth_inp = (depth_inp - self.mean) / self.std
    #     disparities = self.depth(depth_inp)
    #
    #     resized_disparities = [
    #         f.interpolate(disp,
    #                       size=(self.input_height, self.input_width),
    #                       mode="bilinear",
    #                       align_corners=False)
    #         for disp in disparities
    #     ]
    #     depths = [self.disp_to_depth(disp)[-1] for disp in resized_disparities]
    #
    #     synthetic_targets = self.get_synthetic(nested_inp, frame_ids)
    #     identity_losses = list()
    #     for view, _, _ in synthetic_targets:
    #         identity_losses.append(self.project_loss(view, raw_view))
    #
    #     identity_losses = torch.stack(identity_losses, dim=1)
    #
    #     recon_loss = 0.0
    #     smooth_loss = 0.0
    #
    #     for s in self.scales:
    #         projected_losses = list()
    #         depth, disp = depths[s], disparities[s]
    #         for view, r, t in synthetic_targets:
    #             synthetic_view = self.synthetic_from_view(depth, view, r, t, k, inv_k)
    #             projected_losses.append(self.project_loss(synthetic_view, raw_view))
    #         noise = torch.randn(identity_losses.shape, device=identity_losses.device) * 0.00001
    #         projected_losses = torch.stack(projected_losses, dim=1)
    #         valid_loss, _ = torch.min(torch.cat([identity_losses + noise, projected_losses], dim=1), dim=1)
    #         valid_loss = valid_loss.mean()
    #         smooth_loss += 0.001 * self.smooth_loss(disp, nested_inp['raw', 0, s]) / (2 ** s)
    #         recon_loss += valid_loss
    #
    #     total_loss = (recon_loss + smooth_loss) / len(self.scales)
    #     return total_loss

    def get_synthetic(self, nested_inp, frame_ids):
        synthetic_targets = list()
        if self.strategy == Strategy.MONO_ONLY:
            synthetic_targets.extend(self.get_mono_synthetic(nested_inp, frame_ids))
        elif self.strategy == Strategy.STEREO_ONLY:
            synthetic_targets.extend(self.get_stereo_synthetic(nested_inp, frame_ids))
        elif self.strategy == Strategy.BOTH:
            synthetic_targets.extend(self.get_mono_synthetic(nested_inp, frame_ids))
            synthetic_targets.extend(self.get_stereo_synthetic(nested_inp, frame_ids))
        else:
            raise ValueError("unsupported strategy!")
        return synthetic_targets

    def get_mono_synthetic(self, nested_inp, frame_ids):
        raw_p = nested_inp[('raw', frame_ids[0], 0)]
        raw_a = nested_inp[('raw', frame_ids[2], 0)]
        aug_p = nested_inp[('aug', frame_ids[0], 0)]
        aug_c = nested_inp[('aug', frame_ids[1], 0)]
        aug_a = nested_inp[('aug', frame_ids[2], 0)]

        pose_inp_pc = (torch.cat([aug_p, aug_c], dim=1) - self.pair_mean) / self.pair_std
        pose_inp_ca = (torch.cat([aug_c, aug_a], dim=1) - self.pair_mean) / self.pair_std
        r_pc, t_pc = self.pose(pose_inp_pc)
        r_pc, t_pc = transformation_parameters(r_pc, t_pc, invert=False)

        r_ca, t_ca = self.pose(pose_inp_ca)
        r_ac, t_ac = transformation_parameters(r_ca, t_ca, invert=True)

        return [(raw_p, r_pc, t_pc), (raw_a, r_ac, t_ac)]

    def get_stereo_synthetic(self, nested_inp, frame_ids):
        assert frame_ids[-1] == 's'
        stereo = nested_inp[("raw", 's', 0)]
        r_sc = nested_inp.get('stereo_rot', None)
        t_sc = nested_inp.get('stereo_trans', None)
        assert r_sc is not None and t_sc is not None
        return [(stereo, r_sc, t_sc)]

    def disp_to_depth(self, disp):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def predict_depth(self, x):
        disparity = self.predict_disparity(x)
        scaled_disp, depth = self.disp_to_depth(disparity)
        return scaled_disp, depth

    def predict_disparity(self, x):
        depth_inp = (x - self.mean) / self.std
        disparity = self.depth(depth_inp)[0]
        return disparity

    def synthetic_from_view(self, depth, source_image, rotation_matrix, translation, k, inv_k):
        """
        :param depth: Tensor(b1hw),depth of current view
        :param source_image: Tensor(b3hw)
        :param rotation_matrix: Tensor(b33)
        :param translation: Tensor(b3)
        :param k: Tensor(b33) camera intrinsic
        :param inv_k: Tensor(b33) inverse camera intrinsic
        :return:
        """
        # homo_mesh: bhw3
        normalized_camera_points = torch.einsum("hwc,brc->bhwr", self.homo_mesh, inv_k)
        # bhw3 * bhw1 = bhw3
        camera_points = normalized_camera_points * depth.permute(0, 2, 3, 1)
        # p = rp + t
        transformed_camera_points = torch.einsum("bhwc,brc->bhwr",
                                                 camera_points,
                                                 rotation_matrix) + translation[:, None, None, :]
        # bhw3
        projected_camera_points = torch.einsum("bhwc,brc->bhwr", transformed_camera_points, k)
        # normalized camera uv
        image_coords = projected_camera_points[..., :2] / (projected_camera_points[..., 2:3] + 1e-6)
        image_coords[..., 0] /= (self.input_width - 1)
        image_coords[..., 1] /= (self.input_height - 1)
        sample_idx = (image_coords - 0.5) * 2
        reconstructed_img = f.grid_sample(source_image, sample_idx, padding_mode="border", align_corners=False)
        return reconstructed_img


def build_model(args, **kwargs):
    strategy_dict = {"M": Strategy.MONO_ONLY, "MS": Strategy.BOTH, "S": Strategy.STEREO_ONLY}
    model = MonoDepth(scales=args.scales,
                      min_depth=args.min_depth,
                      max_depth=args.max_depth,
                      input_height=args.height,
                      input_width=args.width,
                      strategy=strategy_dict[args.strategy])
    return model
