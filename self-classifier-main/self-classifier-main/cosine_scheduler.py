import jax
import jax.numpy as jnp


def cosine_decay_schedule(
    init_value,
    ramped_up_value,
    ramped_up_epoch,
    total_epochs,
    final_lr,
    steps_per_epoch,
):
    ramped_up_step = ramped_up_epoch * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    @jax.jit
    def schedule(step):
        alpha = (1 / jnp.pi) * jnp.arccos(final_lr / ramped_up_value)
        lr = ramped_up_value * jnp.cos((alpha * jnp.pi * step) / (total_steps))
        lr = jax.lax.cond(
            step <= ramped_up_step,
            lambda: ((ramped_up_value - init_value) / ramped_up_step) * step
            + init_value,
            lambda: lr,
        )
        lr = jax.lax.cond(step > total_steps, lambda: final_lr, lambda: lr)
        return lr

    return schedule
