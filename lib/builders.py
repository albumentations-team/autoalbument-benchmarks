import os

import albumentations as A
import torch
from hydra.utils import instantiate, to_absolute_path
import timm

import segmentation_models_pytorch as smp


def create_model(task, model_cfg, device):
    if "_target_" in model_cfg:
        model = instantiate(model_cfg)
    elif task == "classification":
        model = create_classification_model(model_cfg)
    elif task == "semantic_segmentation":
        model = create_semantic_segmentation_model(model_cfg)
    else:
        raise ValueError(f"Unsupported task {task}")
    model = model.to(device)
    return model


def create_classification_model(model_cfg):
    model = timm.create_model(
        model_cfg.architecture, pretrained=model_cfg.pretrained, num_classes=model_cfg.num_classes
    )
    return model


def create_semantic_segmentation_model(model_cfg):
    model = getattr(smp, model_cfg.architecture)
    encoder_weights = "imagenet" if model_cfg.pretrained else None
    return model(model_cfg.encoder_architecture, encoder_weights=encoder_weights, classes=model_cfg.num_classes)


def create_criterion(criterion_cfg, device):
    criterion = instantiate(criterion_cfg)
    criterion = criterion.to(device)
    return criterion


def create_optimizer(optimizer_cfg, model_params):
    optimizer = instantiate(optimizer_cfg, params=model_params)
    return optimizer


def create_albumentations_transform(albumentations_config_file, root_code_directory):
    absolute_albumentations_config_filepath = os.path.join(
        root_code_directory, "albumentations_configs", albumentations_config_file
    )
    transform = A.load(absolute_albumentations_config_filepath)
    return transform


def create_dataset(dataset_cfg, transform):
    dataset = instantiate(dataset_cfg, transform=transform)
    return dataset


def create_dataloader(dataloader_cfg, dataset, sampler=None):
    dataloader = instantiate(
        dataloader_cfg,
        dataset=dataset,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    return dataloader


def create_device(device, index):
    if index is not None:
        return torch.device(index)
    return torch.device(device)


def create_scheduler(scheduler_cfg, optimizer):
    if scheduler_cfg is None:
        return None
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
    return scheduler
