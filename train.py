import argparse
import collections
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    G_loss_module = config.init_obj(config["G_loss"], module_loss).to(device)
    D_loss_module = config.init_obj(config["D_loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    G_params = model.generator.parameters()
    D_params = model.descriminator.parameters()
    G_optimizer = config.init_obj(config["G_optimizer"], torch.optim, G_params)
    lr_G_scheduler = config.init_obj(config["lr_G_scheduler"], torch.optim.lr_scheduler, G_optimizer)
    D_optimizer = config.init_obj(config["D_optimizer"], torch.optim, D_params)
    lr_D_scheduler = config.init_obj(config["lr_D_scheduler"], torch.optim.lr_scheduler, D_optimizer)

    trainer = Trainer(
        model,
        G_loss_module,
        D_loss_module,
        G_optimizer,
        D_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_G_scheduler=lr_G_scheduler,
        lr_D_scheduler=lr_D_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint to resume training from it (default: None)",
    )
    args.add_argument(
        "-p",
        "--pretrained",
        default=None,
        type=str,
        help="path to latest checkpoint to init model weights with it (default: None)",
    )

    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
