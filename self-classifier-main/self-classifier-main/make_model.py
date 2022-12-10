import jax_resnet
import flax.linen as nn
from flax.linen.normalization import BatchNorm
from jax import random
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
from typing import List, Tuple


class MLP(nn.Module):
    backbone: nn.Sequential

    @nn.compact
    def __call__(self, x, train: bool):
        x = self.backbone(x)
        x = nn.Dense(features=4096)(x)
        x = BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        return nn.Dense(features=128)(x)


class Heads(nn.Module):
    mlp: MLP
    n_classes: List[int]

    @nn.compact
    def __call__(self, x, train: bool):
        x = self.mlp(x, train)
        return tuple(
            [
                nn.Sequential(
                    [
                        nn.leaky_relu,
                        nn.Dense(features=n, use_bias=False),
                    ]
                )(x)
                for n in self.n_classes
            ]
        )


class LinearHead(nn.Module):
    backbone: nn.Sequential
    n_classes: int

    def setup(self):
        self.head = nn.Dense(features=self.n_classes, use_bias=False)

    def __call__(self, x):
        x = self.backbone(x)
        return self.head(x)


def get_backbone():
    # n_classes doesn't matter, we remove the only layer impacted by it anyway.
    # But we need to pass it to construct a ResNet.
    base_model = jax_resnet.ResNet18(n_classes=2)
    backbone = nn.Sequential(base_model.layers[:-1])
    return backbone


def make_pretrain_net(
    key: random.KeyArray, n_classes: List[int], image_size: Tuple[int, int, int]
) -> Tuple[Heads, FrozenDict]:
    dummy_input = np.zeros((1,) + image_size)

    backbone = get_backbone()
    key, param_key1 = random.split(key)
    backbone_params = backbone.init(rngs=param_key1, x=dummy_input)

    heads = Heads(mlp=MLP(backbone), n_classes=n_classes)
    key, param_key2 = random.split(key)
    heads_params = heads.init(rngs=param_key2, x=dummy_input, train=False)

    heads_params = unfreeze(heads_params)
    heads_params["params"]["mlp"]["backbone"] = backbone_params["params"]
    heads_params["batch_stats"]["mlp"]["backbone"] = backbone_params[
        "batch_stats"
    ]
    heads_params = freeze(heads_params)

    return heads, heads_params


def make_linear_net(
    key: random.KeyArray,
    pretrain_state,
    n_classes: int,
    image_size: Tuple[int, int, int],
) -> Tuple[LinearHead, FrozenDict]:
    dummy_input = np.zeros((1,) + image_size)

    backbone = get_backbone()
    key, param_key1 = random.split(key)
    backbone.init(rngs=param_key1, x=dummy_input)

    linear_head = LinearHead(backbone=backbone, n_classes=n_classes)
    key, param_key2 = random.split(key)
    linear_params = linear_head.init(rngs=param_key2, x=dummy_input)

    linear_params = unfreeze(linear_params)
    linear_params["params"]["backbone"] = pretrain_state.params["mlp"][
        "backbone"
    ]
    linear_params["batch_stats"]["backbone"] = pretrain_state.batch_stats[
        "mlp"
    ]["backbone"]
    linear_params = freeze(linear_params)

    return linear_head, linear_params
