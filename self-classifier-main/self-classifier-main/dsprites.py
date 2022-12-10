"""Web-to-disk model."""

import torch
from torchvision.datasets.utils import check_integrity, download_url
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from dataclasses import dataclass
import os
from typing import Callable, Dict
from abc import abstractmethod


@dataclass
class Resource:
    """
    Represents a downloadable web resource.
    """

    filename: str
    url: str
    md5: str

    def validate(self, base_path):
        """
        Validate that the resource exists.

        Params:
            base_path: The directory in which the resource should exist.

        Returns:
            A boolean indicating whether the resource exists and passes the
            integrity check.
        """
        file_path = os.path.join(base_path, self.filename)
        if not os.path.exists(file_path):
            return False
        if not self._validate_md5(file_path, self.md5):
            return False

        return True

    def _validate_md5(self, filepath, md5):
        return check_integrity(filepath, md5)

    def download(self, base_path):
        """
        Download this resource.

        Params:
            base_path: The directory to download the resource into.
        """
        download_url(self.url, base_path, self.filename, self.md5)


class BaseDisentanglementDataset(Dataset):
    """
    Handles the downloading and verification of resources, and transforming
    elements of the dataset.
    """

    # Downloadable resources. Maps the name of the resource to the Resource
    # object.
    resources = {}

    def __init__(
        self, root: str, transform: Callable = None, download: bool = False
    ) -> None:
        """
        Params:
            root: The root directory of all datasets.
            transform: A Callable that will transform each item in the dataset
                       when indexed.
            download: Whether to download the dataset if it's not available in
                      `root`.
        """
        self.root = root
        self.transform = transform or (lambda x: x)

        # Validate resources, downloading them if necessary.
        self.validate_resources(download=download)

    @abstractmethod
    def length(self):
        """Length of the dataset, should be overridden in child."""

    @abstractmethod
    def get_item(self, idx):
        """Get the `idx`th item of the dataset, should be overridden in child."""

    def validate_resources(self, download: bool) -> None:
        """
        Validate the existence and integrity of resources.

        Params:
            download: Whether to download the resources if they don't exist or
                      the integrity check fails.
        """
        for resource in self.resources.values():
            if not resource.validate(self.path):
                if download:
                    resource.download(self.path)
                else:
                    raise RuntimeError(
                        "Resource for dataset not available, pass download=True"
                    )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Gets and transforms the item.

        Always returns a dictionary mapping strings to tensors.

        Params:
            idx: The index in the dataset, must be less than the length.

        Returns:
            item: The transformed element of the dataset.
        """
        item = self.get_item(idx)
        return (self.transform(item[0]), item[1])

    def __len__(self) -> int:
        """The length of the dataset."""
        return self.length()

    def resource_path(self, resource_name):
        """The path to the resource with the given name on disk."""
        return os.path.join(self.path, self.resources[resource_name].filename)

    @property
    def path(self):
        """The path to this dataset's resources."""
        return os.path.join(self.root, self.__class__.__name__)


class DSprites(BaseDisentanglementDataset):
    """
    DSprites is a dataset designed for evaluating disentanglement models. It
    consists of three shapes which vary in position and rotation. The first
    latent factor, color, is constant (white).

    [1] https://github.com/deepmind/dsprites-dataset
    """

    # The entire dataset is offered in a numpy zip archive on GitHub.
    resources = {
        "dataset": Resource(
            filename="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            url="https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true",
            md5="7da33b31b13a06f4b04a70402ce90c2e",
        )
    }

    shapes = (64, 64)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.load_dataset()

    def load_dataset(self):
        """
        Load the numpy archive and convert to torch tensors.
        """
        raw = np.load(self.resource_path("dataset"))
        self.images = raw["imgs"]
        self.latent_factors = raw["latents_values"]

    def length(self):
        """Number of images."""
        return self.images.shape[0]

    def get_item(self, idx):
        img = self.images[idx, :, :]
        return Image.fromarray(img, mode="L"), self.latent_factors[idx, :]
