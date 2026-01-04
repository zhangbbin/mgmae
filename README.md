# MGMAE: Medical Image Segmentation with 3D Masked Autoencoders

这是一个用于医学图像分析（特别是分割任务）的 3D Masked Autoencoder (MAE) 的 PyTorch 实现。该项目利用 Vision Transformer (ViT) 架构，通过掩码自编码器的方式进行自监督预训练，以提取 3D 医学图像（如 CT/MRI）的特征。

## ✨ 主要特性

* **3D MAE 预训练**: 支持对 3D 医疗体数据进行 Masked Autoencoder 预训练。
* **Vision Transformer (ViT) 骨干**: 使用 ViT 作为编码器和解码器的基础架构。
* **分布式训练**: 支持单机多卡 (DDP) 和多机多卡训练。
* **可视化与监控**: 集成 [WandB](https://wandb.ai/) (Weights & Biases) 用于实时监控训练损失、重建效果（PSNR, SSIM）以及可视化掩码重建图像。
* **灵活配置**: 基于 YAML 文件和 `OmegaConf` 进行参数配置。

## 🛠️ 环境依赖

请确保您的环境中安装了以下依赖库：

* Python 3.x
* [PyTorch](https://pytorch.org/) (建议配合 CUDA 使用)
* [WandB](https://docs.wandb.ai/)
* OmegaConf
* Nibabel
* Timm
* Numpy
* Matplotlib
* Scikit-learn

可以通过以下命令安装主要依赖（示例）：

```bash
pip install torch torchvision torchaudio
pip install wandb omegaconf nibabel timm scikit-learn matplotlib
