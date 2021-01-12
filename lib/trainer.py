import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm

from lib.builders import (
    create_model,
    create_criterion,
    create_optimizer,
    create_dataset,
    create_dataloader,
    create_device,
    create_albumentations_transform,
    create_scheduler,
)
from lib.distributed import check_if_main_worker
from lib.loggers import get_csv_logger
from lib.metrics import BestResultTracker, get_metric_tracker
from lib.utils import set_seed


def main(cfg, root_code_directory):
    if cfg.distributed.use_distributed:
        os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
        os.environ["MASTER_PORT"] = str(cfg.distributed.master_port)
        mp.spawn(main_worker, nprocs=cfg.distributed.num_gpus, args=(cfg, root_code_directory))
    else:
        main_worker(None, cfg, root_code_directory)


def main_worker(rank, cfg, root_code_directory):
    cudnn.benchmark = cfg.performance.cudnn_benchmark
    cudnn.deterministic = cfg.performance.cudnn_deterministic
    set_seed(cfg.seed)
    use_distributed = cfg.distributed.use_distributed
    if use_distributed:
        dist.init_process_group(
            backend=cfg.distributed.backend, init_method="env://", world_size=cfg.distributed.num_gpus, rank=rank
        )

    if cfg.save.save_best or cfg.save.save_each_n_epochs:
        os.makedirs("checkpoints", exist_ok=True)

    device = create_device(cfg.device.device, rank)

    model = create_model(cfg.task, cfg.model, device)

    train_transform = create_albumentations_transform(cfg.albumentations.train_config_file, root_code_directory)
    val_transform = create_albumentations_transform(cfg.albumentations.val_config_file, root_code_directory)

    train_dataset = create_dataset(cfg.train_dataset, transform=train_transform)
    val_dataset = create_dataset(cfg.val_dataset, transform=val_transform)

    train_sampler = None
    if use_distributed:
        num_gpus = cfg.distributed.num_gpus
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_gpus, rank=rank
        )
    train_loader = create_dataloader(cfg.train_dataloader, train_dataset, sampler=train_sampler)
    val_loader = create_dataloader(cfg.val_dataloader, val_dataset)

    criterion = create_criterion(cfg.criterion, device)
    optimizer = create_optimizer(cfg.optim, model.parameters())
    scheduler = create_scheduler(getattr(cfg, "scheduler", None), optimizer)
    is_main_worker = check_if_main_worker(use_distributed, rank)
    if is_main_worker:
        best_result_tracker = BestResultTracker(cfg.task)
        csv_logger = get_csv_logger(cfg.task, cfg.logging.csv_file)
    else:
        best_result_tracker = None
        csv_logger = None

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.performance.enable_amp)

    for epoch in range(1, cfg.epochs + 1):
        if use_distributed:
            train_sampler.set_epoch(epoch)
        train_metrics = train(
            train_loader, model, criterion, optimizer, scheduler, epoch, cfg, device, rank, scaler, use_distributed
        )
        if not is_main_worker:
            continue
        val_metrics = validate(val_loader, model, criterion, device, epoch, cfg)
        csv_logger.write_metrics(epoch, train_metrics, val_metrics)

        is_best = best_result_tracker.is_best(val_metrics)
        if is_best:
            best_value = best_result_tracker.get_best()
            metric_name = best_result_tracker.metric_name
            print(f"Best {metric_name} so far: {best_value:.2f}")
        state = {
            "epoch": epoch,
            "cfg": cfg,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cfg.save.save_each_n_epochs and epoch % cfg.save.save_each_n_epochs:
            torch.save(state, os.path.join("checkpoints", f"epoch_{epoch}.pth.tar"))

        if cfg.save.save_best and is_best:
            torch.save(state, os.path.join("checkpoints", "best.pth.tar"))
    if is_main_worker:
        print("Best accuracy:", best_result_tracker.get_best())


def forward_pass(images, target, model, criterion, device, metric_tracker, cfg):
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=cfg.performance.enable_amp):
        output = model(images)
        loss = criterion(output, target)
    if metric_tracker:
        batch_size = images.size(0)
        metric_tracker.update_metrics(output, target, loss, batch_size)
    return loss


def train(train_loader, model, criterion, optimizer, scheduler, epoch, cfg, device, rank, scaler, use_distributed):
    model.train()
    if check_if_main_worker(use_distributed, rank):
        progress_bar = tqdm(total=len(train_loader))
        metric_tracker = get_metric_tracker(cfg.task)
    else:
        progress_bar = None
        metric_tracker = None
    for (images, target) in train_loader:
        optimizer.zero_grad()
        loss = forward_pass(images, target, model, criterion, device, metric_tracker, cfg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if progress_bar:
            progress_bar.set_description(f"Epoch {epoch}. Train. {metric_tracker}")
            progress_bar.update()
    if progress_bar:
        progress_bar.close()
    if scheduler:
        scheduler.step()
    return metric_tracker.get_avg_values() if metric_tracker else None


def validate(val_loader, model, criterion, device, epoch, cfg):
    model.eval()
    metric_tracker = get_metric_tracker(cfg.task)
    progress_bar = tqdm(total=len(val_loader))
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            forward_pass(images, target, model, criterion, device, metric_tracker, cfg)
            progress_bar.set_description(f"Epoch {epoch}. Val. {metric_tracker}")
            progress_bar.update()
    progress_bar.close()
    return metric_tracker.get_avg_values()
