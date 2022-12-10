import optax
import flax
from typing import Optional
from flax.core.frozen_dict import freeze
from flax.training.checkpoints import restore_checkpoint
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from hyperparams import PretrainHyperParams, LinearHyperParams
from make_model import make_pretrain_net, make_linear_net
from cosine_scheduler import cosine_decay_schedule
from get_dataset import dataset_shape


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    epoch: int


def get_pretrain_state(
    hp: PretrainHyperParams, steps_per_epoch: int, *, minimum_epoch: int = 0
) -> Optional[TrainState]:
    heads, heads_state = make_pretrain_net(
        PRNGKey(hp.seed), hp.n_classes, dataset_shape(hp.dataset)
    )
    tx = optax.lars(
        cosine_decay_schedule(
            hp.lr_init_value,
            hp.lr_top_value,
            hp.lr_top_epoch,
            hp.num_epochs,
            hp.lr_final_value,
            steps_per_epoch,
        ),
        weight_decay=1e-6,
    )
    pretrain_state = TrainState.create(
        apply_fn=heads.apply,
        params=heads_state["params"],
        tx=tx,
        batch_stats=heads_state["batch_stats"],
        epoch=0,
    )
    pretrain_state = restore_checkpoint(
        "checkpoints/", prefix=hp.ckpt_prefix(), target=pretrain_state
    )
    if pretrain_state.epoch < minimum_epoch:
        return None
    print("Loaded pretrained model with epoch", pretrain_state.epoch)
    return pretrain_state


def zero_grads():
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


def get_lineval_state(
    lp: LinearHyperParams, steps_per_epoch: int, *, minimum_epoch: int = 0
) -> Optional[TrainState]:

    pretrain_state = get_pretrain_state(
        lp.pretrain_params,
        steps_per_epoch,
        minimum_epoch=lp.pretrain_params.num_epochs,
    )
    assert pretrain_state is not None, "you must pretrain first"

    linear_head, linear_state = make_linear_net(
        PRNGKey(lp.seed),
        pretrain_state,
        lp.n_classes,
        dataset_shape(lp.dataset),
    )
    tx = optax.multi_transform(
        {
            "lars": optax.lars(
                cosine_decay_schedule(
                    lp.lr_init_value,
                    lp.lr_top_value,
                    lp.lr_top_epoch,
                    lp.num_epochs,
                    lp.lr_final_value,
                    steps_per_epoch,
                )
            ),
            "zero": zero_grads(),
        },
        freeze({"backbone": "zero", "head": "lars"}),
    )
    linear_state = TrainState.create(
        apply_fn=linear_head.apply,
        params=linear_state["params"],
        tx=tx,
        batch_stats=linear_state["batch_stats"],
        epoch=0,
    )
    linear_state = restore_checkpoint(
        "checkpoints/", prefix=lp.ckpt_prefix(), target=linear_state
    )
    if linear_state.epoch < minimum_epoch:
        return None
    print("Loaded linear eval model with epoch", linear_state.epoch)
    return linear_state
