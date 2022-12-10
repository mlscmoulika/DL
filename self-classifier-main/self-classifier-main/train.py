from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from flax.training import checkpoints
import numpy as np
import torch

from train_state import get_pretrain_state, get_lineval_state, TrainState
from data_loader import NumpyLoader
from epoch import pretrain_epoch, lineval_epoch
from hyperparams import LinearHyperParams, PretrainHyperParams
from grid_viz import extract_class_images

import time
from datetime import timedelta

CHECKPOINT_DIR = "checkpoints/"

def pretrain_tb_images(tb_writer, dataset, train_state, step):
    head_figures = extract_class_images(dataset, train_state, double_x=True)
    for head_num, class_figures in enumerate(head_figures):
        class_figures = np.swapaxes(class_figures, 1, -1)
        figure = make_grid(torch.from_numpy(class_figures))
        tb_writer.add_image(f"head_{head_num}", figure, step, dataformats="CWH")


def do_pretraining(
    pretrain_dataset, *, hyperparams: PretrainHyperParams
) -> TrainState:
    dataloader = NumpyLoader(
        pretrain_dataset, batch_size=hyperparams.batch_size
    )
    pretrain_state = get_pretrain_state(
        hyperparams,
        steps_per_epoch=len(pretrain_dataset) // hyperparams.batch_size,
    )
    assert pretrain_state is not None
    epoch = int(pretrain_state.epoch)
    print("Initial epoch", epoch, type(epoch))

    # Tensorboard stuff
    tb_writer = SummaryWriter(
        f"tensorboard_logs/{hyperparams.ckpt_prefix()}",
        filename_suffix="pretrain",
    )

    while epoch < hyperparams.num_epochs:
        start_time = time.perf_counter()

        # Do epoch and save
        pretrain_state, pretrain_loss = pretrain_epoch(
            pretrain_state, dataloader, hyperparams
        )
        epoch += 1
        checkpoints.save_checkpoint(
            ckpt_dir=CHECKPOINT_DIR,
            target=pretrain_state,
            step=epoch,
            prefix=hyperparams.ckpt_prefix(),
        )

        # Tensorboard
        tb_writer.add_scalar("Loss/train", pretrain_loss, epoch)
        if epoch % 4 == 0:
            pretrain_tb_images(tb_writer, pretrain_dataset, pretrain_state, epoch)

        # Timing
        time_elapsed = time.perf_counter() - start_time
        epochs_remaining = hyperparams.num_epochs - epoch
        time_remaining = timedelta(seconds=int(epochs_remaining * time_elapsed))

        print(
            f"Epoch {pretrain_state.epoch}: pretrain loss {pretrain_loss:.3f} ({time_elapsed:.1f} sec, {time_remaining} remaining)"
        )

    return pretrain_state


def do_linear_eval(
    nonaugmented_dataset, *, hyperparams: LinearHyperParams
) -> TrainState:
    dataloader = NumpyLoader(
        nonaugmented_dataset, batch_size=hyperparams.batch_size
    )

    train_state = get_lineval_state(
        hyperparams,
        steps_per_epoch=len(nonaugmented_dataset) // hyperparams.batch_size,
    )
    epoch = int(train_state.epoch)
    print("Initial epoch", epoch, type(epoch))

    while epoch < hyperparams.num_epochs:
        train_state, loss, acc = lineval_epoch(train_state, dataloader)
        epoch += 1
        checkpoints.save_checkpoint(
            ckpt_dir=CHECKPOINT_DIR,
            target=train_state,
            step=epoch,
            prefix=hyperparams.ckpt_prefix(),
        )
        print(
            f"Epoch {train_state.epoch}: supervised loss {loss:.3f} acc {acc:.3f}"
        )

    return train_state
