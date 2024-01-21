import torch
from torch import nn


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0.0, 1.0)


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)

    def __call__(self, disp, img):
        mean_disp = self.mean(disp)
        norm_disp = disp / (mean_disp + 1e-7)
        grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
        grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class ProjectLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(ProjectLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss(reduction="none")
        self.ssim = SSIM()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target).mean(1)
        ssim_loss = self.ssim(pred, target).mean(1)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
