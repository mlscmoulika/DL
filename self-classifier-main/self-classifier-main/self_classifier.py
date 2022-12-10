#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
from tqdm import tqdm

import matplotlib.pyplot as plt
import flax

from flax.training import train_state, checkpoints
import optax
import jax
from jax import random
import jax.numpy as jnp

from flax.core.frozen_dict import freeze, unfreeze


# In[2]:


np.set_printoptions(suppress=True)


pretrain_dataset = torchvision.datasets.CIFAR10(
    "./cifar10",
    download=True,
    transform=TransformsSimCLR(is_pretrain=True, is_val=False),
    train=True,
)

pretrain_dataloader = NumpyLoader(pretrain_dataset, batch_size=128)


# In[7]:


(x1, x2), y = pretrain_dataset[1]
print(
    "Image x1 shape: ",
    x1.shape,
    "\nImage x2 shape: ",
    x2.shape,
    "\nclass index y: ",
    y,
    "\n",
)

merged_images = np.concatenate([x1, x2], axis=1)
plt.imshow(merged_images)

# ## Utils

# In[12]:


# ## Self classifier

# In[13]:


# In[15]:


# In[ ]:


(x1, x2), _ = pretrain_dataset[1]
logits = encoder.apply(
    {
        "params": pretrain_state.params,
        "batch_stats": pretrain_state.batch_stats,
    },
    jnp.array([x1, x2]),
    mutable=False,
)

probs = jax.nn.softmax(logits, axis=1)
print(np.array(probs))


# In[ ]:


imgs = np.array([pretrain_dataset[i][0][0] for i in range(5000)])


# In[ ]:


logits = encoder.apply(
    {
        "params": pretrain_state.params,
        "batch_stats": pretrain_state.batch_stats,
    },
    jnp.array(imgs),
    mutable=False,
)

probs = jax.nn.softmax(logits, axis=1)
classes = jnp.argmax(probs, axis=1)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for class_idx in range(0, 9):
    figure = np.zeros((32 * 3, 32 * 3, 3))
    class_imgs = imgs[classes == class_idx]
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            if class_imgs.shape[0] <= idx:
                img = np.zeros((32, 32, 3))
            else:
                img = class_imgs[idx] / 255
            figure[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32, :] = img
    ax = axes[class_idx // 3, class_idx % 3]
    ax.imshow(figure)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# In[ ]:
