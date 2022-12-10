import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from get_dataset import get_dataset
from hyperparams import load_pretrain_params
from train_state import get_pretrain_state
from grid_viz import extract_class_images


def handle_viz1(args) -> int:
    hyperparams = load_pretrain_params()
    dataset = get_dataset(hyperparams.dataset)["validation"]

    state = get_pretrain_state(
        hyperparams,
        steps_per_epoch=len(dataset) // hyperparams.batch_size,
    )
    assert state is not None

    figures = extract_class_images(dataset, state, double_x=False)[0]

    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(figures[i * 3 + j], cmap='gray')

    plt.savefig("viz1.png")

    return 0
