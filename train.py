#!/usr/bin/env python3
"""Train primary or reference diffusion segmentation models."""

import json
import os
import random
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataset.source_validation_augmentation import SourceValidationAugmentationDataset, load_conditions
from model.ddpm import CategoricalDiffusion, Trainer
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
import importlib
from model.denoisers import TimeAwareSwinUNETR


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _build_source_validation_loaders(cfg: DictConfig, dataset_module, val_root: str):
    manifest_path = PACKAGE_ROOT / "metadata" / "card_source_val_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in manifest["datasets"]["acdc"]["seeds"]]
    conditions = list(load_conditions())
    if len(conditions) != len(seeds):
        raise ValueError("Source-validation conditions and manifest seeds differ")

    split_config = _resolve_to_dict(cfg.dataset.get("split_config")) or {}
    split_config["roi_x"] = cfg.dataset.get("roi_x", 256)
    split_config["roi_y"] = cfg.dataset.get("roi_y", 256)
    split_config["roi_z"] = cfg.dataset.get("roi_z", 1)
    batch_size = int(cfg.dataset.get("test_batch_size", cfg.dataset.get("batch_size", 8)))
    num_workers = int(cfg.dataset.get("val_num_workers", 0))
    num_fg = int(cfg.dataset.get("num_foreground_classes", 3))
    use_bg = bool(cfg.dataset.get("categorical_use_background", True))
    pred_sdf = bool(cfg.dataset.get("predict_background_sdf", False))

    loaders = []
    for condition, cond_seed in zip(conditions, seeds):
        base = dataset_module.ACDC_nnUNet_Dataset(
            root_dir=val_root,
            mode="val",
            stage_two=False,
            num_foreground_classes=num_fg,
            predict_background_sdf=pred_sdf,
            categorical_use_background=use_bg,
            split_config=dict(split_config),
        )
        wrapped = (
            base
            if condition == "clean"
            else SourceValidationAugmentationDataset(base, condition, cond_seed)
        )
        loader = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dataset_module.data_collate,
        )
        loaders.append((condition, loader))
    return loaders


def _resolve_to_dict(value):
    """Convert optional DictConfig to a plain dictionary."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def require_cuda_device(gpu_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    torch.cuda.init()
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def _get_swinunetr_config(swin_type: str = 'base'):
    """Get SwinUNETR configuration based on model size."""
    configs = {
        'mini': {
            'feature_size': 12,
            'depths': [1, 1, 1, 1],
            'num_heads': [3, 6, 12, 24],
        },
        'tiny': {
            'feature_size': 12,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
        },
        'small': {
            'feature_size': 24,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
        },
        'base': {
            'feature_size': 48,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
        },
        'large': {
            'feature_size': 60,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
        }
    }
    return configs.get(swin_type, configs['base'])


class MONAIModelWrapper(torch.nn.Module):
    """Wrapper for MONAI models to match the expected interface."""

    def __init__(self, model, *, cond_channels: int = 0):
        super().__init__()
        self.model = model
        self.cond_channels = int(cond_channels)

    def forward(self, x, t, **kwargs):
        output = self.model(x)
        if isinstance(output, list):
            return output[-1]
        return output

    def forward_with_cond_scale(self, x, t, *, cond_scale: float = 1.0, **kwargs):
        logits = self.forward(x, t, **kwargs)
        if cond_scale == 1.0 or self.cond_channels <= 0:
            return logits
        if x.shape[1] < self.cond_channels:
            raise ValueError('Input tensor has fewer channels than configured conditioning channels')
        uncond_x = x.clone()
        uncond_x[:, -self.cond_channels:, ...] = 0
        null_logits = self.forward(uncond_x, t, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale


def _ensure_cfg_wrapper(model: torch.nn.Module, cond_channels: int) -> torch.nn.Module:
    if hasattr(model, 'forward_with_cond_scale'):
        return model
    return MONAIModelWrapper(model, cond_channels=cond_channels)


def create_denoising_model(cfg: DictConfig, dataset_cfg: DictConfig):
    """Create a denoising backbone model based on configuration.

    Args:
        cfg: Full configuration (must contain cfg.model).
        dataset_cfg: Dataset configuration.

    Returns:
        A model instance with `forward_with_cond_scale` method.
    """
    name = str(cfg.model.denoising_fn).lower()
    if name not in {"swinunetr", "swin_unetr"}:
        raise ValueError("This package trains only the cardiac SwinUNETR denoiser")
    total_in = cfg.model.total_input_channels
    out_channels = cfg.model.diffusion_num_channels
    swin_type = cfg.model.get('swinunetr_type', 'base')
    defaults = {
        'spatial_dims': 2,
        'in_channels': total_in,
        'out_channels': out_channels,
        'patch_size': 2,
        'window_size': [7, 7],
        'use_checkpoint': False,
        'use_v2': True,
        'cond_channels': cfg.model.cond_channels,
        **_get_swinunetr_config(swin_type),
    }
    defaults.update(_resolve_to_dict(cfg.model.get('swinunetr_kwargs', {})) or {})
    model = TimeAwareSwinUNETR(**defaults)

    return _ensure_cfg_wrapper(model, cfg.model.cond_channels)


@hydra.main(config_path='config', config_name='joint_cfg', version_base=None)
def run(cfg: DictConfig):
    """Main training entry point."""
    with open_dict(cfg):
        cfg.model.gpus = 0
        cfg.model.save_and_sample_every = cfg.model.get('save_and_sample_every', 250)
        cfg.model.gradient_accumulate_every = cfg.model.get('gradient_accumulate_every', 1)

        if 'gpus_override' in cfg.model:
            cfg.model.gpus = cfg.model.gpus_override
        if 'train_num_steps_override' in cfg.model:
            cfg.model.train_num_steps = cfg.model.train_num_steps_override
        if 'save_and_sample_every_override' in cfg.model:
            cfg.model.save_and_sample_every = cfg.model.save_and_sample_every_override
        if 'results_folder_postfix_override' in cfg.model:
            cfg.model.results_folder_postfix = cfg.model.results_folder_postfix_override
        if 'tensorboard_dir_override' in cfg.model:
            cfg.model.tensorboard_dir = cfg.model.tensorboard_dir_override

        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

        os.makedirs(cfg.model.results_folder, exist_ok=True)
        config_save_path = os.path.join(cfg.model.results_folder, 'config.yaml')
        with open(config_save_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print(f"Configuration saved to: {config_save_path}")

        if cfg.model.diffusion_type != "categorical":
            raise ValueError("This package supports categorical diffusion only")

        down_factor = max(1, cfg.dataset.get('pixel_unshuffle_factor', 1))
        image_channels = cfg.dataset.get('image_channels', 1)
        mask_classes = cfg.dataset.get('mask_classes', 0)

        if mask_classes <= 0:
            raise ValueError("cfg.dataset.mask_classes must be a positive integer")

        spatial_dims = cfg.get('spatial_dims', 3)
        channels_per_voxel = down_factor ** spatial_dims
        cond_channels = image_channels * channels_per_voxel
        cfg.model.cond_channels = cond_channels

        categorical_use_background = bool(cfg.dataset.get('categorical_use_background', True))
        cfg.dataset.categorical_use_background = categorical_use_background
        cfg.model.categorical_final_step_mode = cfg.model.get('categorical_final_step_mode', 'majority')
        categorical_foreground_classes = int(cfg.dataset.get('num_foreground_classes', mask_classes))
        categorical_num_classes = categorical_foreground_classes + (1 if categorical_use_background else 0)
        cfg.dataset.num_foreground_classes = categorical_foreground_classes
        cfg.model.categorical_num_classes = categorical_num_classes
        cfg.model.categorical_use_background = categorical_use_background
        cfg.model.diffusion_num_channels = categorical_num_classes
        cfg.model.total_input_channels = categorical_num_classes + cond_channels
        cfg.model.out_dim = categorical_num_classes
        cfg.model.objective = 'categorical'

    device = require_cuda_device(cfg.model.gpus)
    print(f"Using CUDA device: {cfg.model.gpus}")
    if cfg.model.get('seed') is not None:
        seed_all(int(cfg.model.seed))

    # Initialize model and diffusion process
    model = create_denoising_model(cfg, cfg.dataset).to(device)
    print(f"Using denoising backbone: {cfg.model.denoising_fn}")

    aux_loss_type = cfg.model.get('categorical_aux_loss_type', 'none')
    aux_loss_weight = cfg.model.get('categorical_aux_loss_weight', 0.0)
    aux_dice_weight = cfg.model.get('categorical_aux_dice_weight', 1.0)
    aux_ce_weight = cfg.model.get('categorical_aux_ce_weight', 1.0)
    aux_boundary_weight = cfg.model.get('categorical_aux_boundary_weight', 0.1)
    kl_loss_weight = cfg.model.get('categorical_kl_loss_weight', 1.0)
    condition_mask_ratio = cfg.model.get('cfg_condition_mask_ratio', 0.1)
    sample_mode = cfg.model.get('sample_mode', 'one_hot')
    train_stochastic = cfg.model.get('train_stochastic', True)
    inference_stochastic = cfg.model.get('inference_stochastic', False)
    noise_std = cfg.model.get('noise_std', 0.0)

    diffusion = CategoricalDiffusion(
        model,
        image_size=cfg.dataset.diffusion_img_size,
        num_frames=cfg.dataset.diffusion_depth_size,
        num_classes=cfg.model.diffusion_num_channels,
        cond_channels=cfg.model.cond_channels,
        timesteps=cfg.model.timesteps,
        objective=cfg.model.get('objective', 'categorical'),
        aux_loss_type=aux_loss_type,
        aux_loss_weight=aux_loss_weight,
        aux_dice_weight=aux_dice_weight,
        aux_ce_weight=aux_ce_weight,
        aux_boundary_weight=aux_boundary_weight,
        kl_loss_weight=kl_loss_weight,
        final_step_mode=cfg.model.categorical_final_step_mode,
        condition_mask_ratio=condition_mask_ratio,
        sample_mode=sample_mode,
        train_stochastic=train_stochastic,
        inference_stochastic=inference_stochastic,
        noise_std=noise_std,
    ).to(device)

    # Load dataset
    dataset = importlib.import_module(f'dataset.{cfg.dataset.name}_dataloader')
    train_dataloader = dataset.get_loader(cfg.dataset, mode='train', stage_two=False)
    val_root_override = cfg.dataset.get('val_root_dir')
    val_dataloader = None
    source_val_loaders = None
    if val_root_override:
        if bool(cfg.model.get('source_val_four_conditions', True)):
            source_val_loaders = _build_source_validation_loaders(cfg, dataset, val_root_override)
        else:
            val_dataloader = dataset.get_loader(
                cfg.dataset, mode='val', root_override=val_root_override
            )

    # Create trainer
    lr_schedule_total_steps = cfg.model.get('lr_schedule_total_steps', cfg.model.train_num_steps)
    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        validation_loader=val_dataloader,
        source_validation_loaders=source_val_loaders,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        lr_schedule_total_steps=lr_schedule_total_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        results_folder=cfg.model.results_folder,
        lr_decay=True,
        log_dir=cfg.model.get('tensorboard_dir'),
        max_grad_norm=cfg.model.get('max_grad_norm', 2.0),
        cfg_enabled=True,
        p_uncond=condition_mask_ratio,
        early_stopping_enabled=bool(cfg.model.get('early_stopping_enabled', False)),
        early_stopping_patience=int(cfg.model.get('early_stopping_patience', 10)),
        early_stopping_min_delta=float(cfg.model.get('early_stopping_min_delta', 0.001)),
        early_stopping_min_updates=int(cfg.model.get('early_stopping_min_updates', 25000)),
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location='cpu', strict=False)
        if cfg.model.get('reset_scheduler_on_load', False) and trainer.scheduler is not None:
            trainer.scheduler.total_steps = int(lr_schedule_total_steps)
            trainer.scheduler.last_step = trainer.step
            lrs = trainer.scheduler.get_lr()
            for param_group, lr in zip(trainer.opt.param_groups, lrs):
                param_group['lr'] = lr
            print(
                "Reset scheduler after checkpoint load: "
                f"step={trainer.step}, total_steps={trainer.scheduler.total_steps}, "
                f"lr={lrs[0]:.2e}"
            )

    trainer.train()


if __name__ == '__main__':
    run()
