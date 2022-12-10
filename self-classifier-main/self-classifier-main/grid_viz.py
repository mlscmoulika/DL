import jax
import jax.numpy as jnp
import numpy as np

def extract_class_images(dataset, train_state, *, double_x: bool):
    dataset_size = len(dataset)
    idxs = np.random.choice(dataset_size, 5000)
    if double_x:
        imgs = jnp.array([dataset[idx][0][0] for idx in idxs])
    else:
        imgs = jnp.array([dataset[idx][0] for idx in idxs])
    head_logits = train_state.apply_fn(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        jnp.array(imgs),
        train=False,
        mutable=False,
    )

    result = []
    for logits in head_logits:
        head_size = logits.shape[1]

        probs = jax.nn.softmax(logits, axis=1)
        classes = np.argmax(probs, axis=1)

        figures = []
        for class_idx in range(0, head_size):
            WIDTH = imgs.shape[1]
            HEIGHT = imgs.shape[2]
            DEPTH = imgs.shape[3]

            figure = np.zeros((WIDTH * 3, HEIGHT * 3, DEPTH))
            class_imgs = imgs[classes == class_idx]
            for i in range(3):
                for j in range(3):
                    idx = 3 * i + j
                    if class_imgs.shape[0] <= idx:
                        img = np.zeros((WIDTH, HEIGHT, DEPTH))
                    else:
                        img = class_imgs[idx] / 255
                    figure[
                        i * WIDTH : (i + 1) * WIDTH,
                        j * HEIGHT : (j + 1) * HEIGHT,
                        :,
                    ] = img
            print("Figure", figure.shape, "logits", logits.shape)
            figures.append(figure)
        result.append(np.array(figures))
    return result