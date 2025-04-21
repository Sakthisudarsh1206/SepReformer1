# utils/util_engine.py
import os
import torch
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np

def load_last_checkpoint_n_get_epoch(
    checkpoint_dir, model, optimizer, location='cpu'
):
    """
    Load the latest checkpoint from a directory.
    Returns the next epoch to start from.
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]
    if not checkpoint_files:
        return 1

    epochs = [int(f.split('.')[1]) for f in checkpoint_files]
    latest = checkpoint_files[epochs.index(max(epochs))]
    latest_path = os.path.join(checkpoint_dir, latest)

    logger.info(f"Loaded Pretrained model from {latest_path} .....")
    ckpt = torch.load(latest_path, map_location=location)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'] + 1

def save_checkpoint_per_nth(
    nth, epoch, model, optimizer,
    train_loss, valid_loss, checkpoint_path
):
    """
    Save model/optimizer state every `nth` epoch.
    """
    if epoch % nth == 0:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            },
            os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
        )

def save_checkpoint_per_best(
    best, valid_loss, train_loss,
    epoch, model, optimizer, checkpoint_path
):
    """
    Save model when validation loss improves.
    Returns the updated best-valid-loss.
    """
    if valid_loss < best:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            },
            os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
        )
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total_parameters += count
        logger.info(f"{name}: {count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input, dummy_input, metrics):
    # ptflops
    if 'ptflops' in metrics:
        macs, params = get_model_complexity_info(
            model,
            (input.shape[1],),
            print_per_layer_stat=False,
            verbose=False
        )
        macs = macs.replace(" MMac", "")
        params = params.replace(" M", "")
        logger.info(f"ptflops: MACs: {macs}, Params: {params}")

    # thop
    if 'thop' in metrics:
        MACs_thop, params_thop = profile(model, inputs=(input,), verbose=False)
        MACs_thop, params_thop = MACs_thop / 1e9, params_thop / 1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")

    # torchinfo
    if 'torchinfo' in metrics:
        profile_ = summary_(model, input_size=input.size(), verbose=0)
        MACs_torchinfo = profile_.total_mult_adds / 1e6
        params_torchinfo = profile_.total_params / 1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")
