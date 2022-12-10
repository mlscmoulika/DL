from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Tuple


@dataclass_json
@dataclass(frozen=True, eq=True)
class PretrainHyperParams:
    dataset: str
    symmetric_loss: bool
    num_epochs: int
    batch_size: int
    n_classes: Tuple[int, ...]
    lr_init_value: float = 0.3
    lr_top_value: float = 4.8
    lr_top_epoch: float = 10
    lr_final_value: float = 4.8e-3
    seed: int = 42

    def ckpt_prefix(self):
        dataset = self.dataset.lower()
        ncls_part = "-".join(map(str, self.n_classes))
        lr_part = f"_la{self.lr_init_value}_lb{self.lr_top_value}_lc{self.lr_top_epoch}_ld{self.lr_final_value}"
        return (
            f"pre_{dataset}_b{self.batch_size}_n{ncls_part}_s{int(self.symmetric_loss)}"
            + lr_part
            + "_epoch"
        )


@dataclass_json
@dataclass
class LinearHyperParams:
    pretrain_params: PretrainHyperParams
    dataset: str
    num_epochs: int
    n_classes: int
    batch_size: int
    lr_init_value: float = 0.3
    lr_top_value: float = 4.8
    lr_top_epoch: float = 10
    lr_final_value: float = 4.8e-3
    seed: int = 42

    def ckpt_prefix(self):
        dataset = self.dataset.lower()
        pretrain_prefix = self.pretrain_params.ckpt_prefix()
        lr_part = f"_la{self.lr_init_value}_lb{self.lr_top_value}_lc{self.lr_top_epoch}_ld{self.lr_final_value}"
        return f"lin_{dataset}_{pretrain_prefix}__n{self.n_classes}_b{self.batch_size}_{lr_part}_epoch"


def load_pretrain_params():
    with open("pretrain.json", "r", encoding="utf-8") as f:
        return PretrainHyperParams.from_json(f.read())


def load_lineval_params():
    with open("lineval.json", "r", encoding="utf-8") as f:
        return LinearHyperParams.from_json(f.read())
