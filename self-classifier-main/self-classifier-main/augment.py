from torchvision import transforms
import numpy as np


def img_to_numpy(img):
    img = np.array(img)
    return np.reshape(img, (img.shape[0], img.shape[1], -1))


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x i and x j, which we consider as a positive pair.
    """

    def __init__(self, is_pretrain=True, is_val=False, needs_grayscale=True):
        self.is_pretrain = is_pretrain
        self.is_val = is_val
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
            ]
            + ([transforms.RandomGrayscale(p=0.2)] if needs_grayscale else [])
            + [transforms.Lambda(img_to_numpy)]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Lambda(img_to_numpy),
            ]
        )

    def __call__(self, x):
        if self.is_pretrain:
            return self.train_transform(x), self.train_transform(x)
        if self.is_val:
            return self.test_transform(x)
        return self.train_transform(x)
