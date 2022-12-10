from loss import someettric_loss
from functools import partial
from hyperparams import PretrainHyperParams

import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

t_col = 0.05
t_row = 0.1


@partial(jax.jit, static_argnames=["symmetric"])
def apply_model_pretrain(train_state, X1, X2, symmetric):
    def loss_fn(params):
        X = jnp.concatenate((X1, X2))
        logits_list, new_state = train_state.apply_fn(
            {
                "params": params,
                "batch_stats": train_state.batch_stats,
            },
            X,
            train=True,
            mutable=["batch_stats"],
        )

        loss_sum = 0.0
        for logits in logits_list:
            logits1, logits2 = jnp.split(logits, 2)
            loss_sum += someettric_loss(
                logits1, logits2, t_row, t_col, symmetric
            )
        loss_avg = loss_sum / len(logits_list)

        return loss_avg, new_state["batch_stats"]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, batch_stats), grads = grad_fn(train_state.params)

    train_state = train_state.apply_gradients(
        grads=grads, batch_stats=batch_stats
    )
    return train_state, loss


def pretrain_epoch(
    train_state, train_dataloader, hyperparams: PretrainHyperParams
):
    batch_loss = []
    train_dataloader_tqdm = tqdm(train_dataloader)
    for ((X1, X2), _) in train_dataloader_tqdm:
        train_state, loss = apply_model_pretrain(
            train_state, X1, X2, hyperparams.symmetric_loss
        )
        batch_loss.append(loss)
        train_dataloader_tqdm.set_postfix({"train_loss": loss.item()})

    train_state = train_state.replace(epoch=train_state.epoch + 1)
    return train_state, np.mean(batch_loss)


@jax.jit
def apply_model_lineval(train_state, X, Y):
    def loss_fn(params):
        logits, new_state = train_state.apply_fn(
            {"params": params, "batch_stats": train_state.batch_stats},
            X,
            mutable=["batch_stats"],
        )

        labels = jax.nn.one_hot(Y, num_classes=logits.shape[1])
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, -1) == Y)
        return loss, (accuracy, new_state["batch_stats"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, batch_stats)), grads = grad_fn(train_state.params)

    train_state = train_state.apply_gradients(
        grads=grads, batch_stats=batch_stats
    )
    return train_state, loss, accuracy


def lineval_epoch(train_state, dataloader):
    batch_loss = []
    batch_acc = []
    dataloader_tqdm = tqdm(dataloader)
    for (X, Y) in dataloader_tqdm:
        train_state, loss, accuracy = apply_model_lineval(train_state, X, Y)
        batch_loss.append(loss)
        batch_acc.append(accuracy)
        dataloader_tqdm.set_postfix(
            {
                "train_loss": loss.item(),
                "train_acc": accuracy.item(),
            }
        )
    train_state = train_state.replace(epoch=train_state.epoch + 1)
    return train_state, np.mean(batch_loss), np.mean(batch_acc)
