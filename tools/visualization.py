import numpy as np

import torch
import torchvision
import pandas as pd

from timm.models.layers.helpers import to_3tuple

import pdb

def calculate_metrics(original, recon):
    """计算PSNR和SSIM指标"""
    # 确保输入形状为 [N, C, H, W]
    assert original.dim() == 4 and recon.dim() == 4

    # 计算PSNR
    mse = torch.mean((original - recon) ** 2, dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))  # 防止除零

    # 计算SSIM（需要torchmetrics）
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        reduction='none'
    ).to(original.device)

    ssim_values = ssim(recon, original)  # 注意输入顺序

    return psnr, ssim_values


def ssim_psnr(image):
    # 从image中提取原始图像和重建图像
    original = image[0]  # 形状 [1, 5, 96, 96] (C, num_slices, H, W)
    recon = image[2]  # 同上

    # save singular
    # 1. 提取第一个 slice 的矩阵
    first_slice = recon[0, 0, :, :]  # 提取 (96, 96) 的矩阵

    # 2. 计算奇异值分解 (SVD)
    _, S, _ = torch.linalg.svd(first_slice)

    # 3. 转换为 DataFrame 并保存
    df = pd.DataFrame(
        S.cpu().numpy(),  # 确保张量在 CPU 并转为 numpy
        columns=["Singular Value"]
    )
    import time
    filename = f"/data/zb/MG-MAE/singular_distribute/svd_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"
    df.to_excel(filename, index=False)

    # 调整维度为 [num_slices, C, H, W]
    original = original.permute(1, 0, 2, 3)  # [5, 1, 96, 96]
    recon = recon.permute(1, 0, 2, 3)  # [5, 1, 96, 96]

    # 计算指标
    psnr_per_group, ssim_per_group = calculate_metrics(original, recon)

    # 计算平均指标
    avg_psnr = psnr_per_group.mean()
    avg_ssim = ssim_per_group.mean()

    # 打印结果
    print(f"Per-group PSNR: {psnr_per_group.cpu().numpy().round(2)}")
    print(f"Per-group SSIM: {ssim_per_group.cpu().numpy().round(4)}")
    print(f"Average PSNR: {avg_psnr.item():.2f} dB")
    print(f"Average SSIM: {avg_ssim.item():.4f}")

    return avg_psnr.item(), avg_ssim.item(), psnr_per_group.cpu().numpy().round(2), ssim_per_group.cpu().numpy().round(4)

def patches3d_to_grid(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, L, C = patches.shape
    patch_size = to_3tuple(patch_size)
    grid_size = to_3tuple(grid_size)
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"
    # print(f"grid_size: {grid_size[0]}, patch_size: {patch_size[0]}")

    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)
    # restore image structure
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0], 
                                                            grid_size[1] * patch_size[1], 
                                                            grid_size[2] * patch_size[2])

    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
        image = torch.cat(slice_list, dim=2)
    else:
        raise ValueError(f"Only support D for now")

    # PSNR AND SSIM
    avg_psnr, avg_ssim, psnr_per_group, ssim_per_group = ssim_psnr(image)

    visH, visW = image.size(-2), image.size(-1)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)
    # grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
    grid_of_images_numpy = grid_of_images.transpose(0, 2).cpu().numpy()

    return grid_of_images_numpy, avg_psnr, avg_ssim, psnr_per_group, ssim_per_group

def images3d_to_grid(image, n_group=3, hidden_axis='d', slice_pos_list=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --image: [B, C, H, W, D]
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, C, H, W, D = image.shape

    assert B % n_group == 0
    n_per_row = B // n_group
    list_of_grid_images = []
    for slice_pos in slice_pos_list:
        if hidden_axis == 'd':
            image_slice = image[..., :, :, int(D * slice_pos)] # [B, 3, H, W]
        else:
            raise ValueError(f"Only support D for now")
        # pdb.set_trace()
        grid_of_images = torchvision.utils.make_grid(image_slice, nrow=n_per_row)
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        list_of_grid_images.append(grid_of_images)

    return list_of_grid_images