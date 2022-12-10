import argparse
from typing import List

from get_dataset import get_dataset
from train import do_pretraining, do_linear_eval
from hyperparams import load_lineval_params, load_pretrain_params
from viz1 import handle_viz1

# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)


def csv_ints(s: str) -> List[int]:
    ints = [int(part.strip()) for part in s.split(",")]
    return ints


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    pretrain_parser = subparsers.add_parser("pretrain")
    lineval_parser = subparsers.add_parser("lineval")
    viz1_parser = subparsers.add_parser("viz1")
    viz2_parser = subparsers.add_parser("viz2")
    return parser.parse_args()


def handle_pretrain(args) -> int:
    hyperparams = load_pretrain_params()
    data_splits = get_dataset(hyperparams.dataset)
    do_pretraining(data_splits["augmented"], hyperparams=hyperparams)
    return 0


def handle_lineval(args) -> int:
    hyperparams = load_lineval_params()
    data_splits = get_dataset(hyperparams.dataset)
    do_linear_eval(data_splits["nonaugmented"], hyperparams=hyperparams)
    return 0


def main() -> int:
    args = parse_args()

    if args.cmd == "pretrain":
        return handle_pretrain(args)
    elif args.cmd == "lineval":
        return handle_lineval(args)
    elif args.cmd == "viz1":
        return handle_viz1(args)
    else:
        raise NotImplementedError(f"Command {args.cmd} not implemented")

    return 1


if __name__ == "__main__":
    exit(main())
