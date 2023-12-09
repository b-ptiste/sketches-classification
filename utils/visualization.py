# import third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets

# import local libraries
from src.data import MixUpCutMixDataset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def pplot_exemple(cfg, transforms_train):
    if cfg["mixup_cutmix_line"] is not None:
        mixup_dataset = MixUpCutMixDataset(
            cfg["data"] + "/train_images",
            transform=transforms_train,
            mix_up_alpha=0.2,
            mix_up_prob=0.5,
            cut_mix_prob=0.5,
            beta=1.0,
            dilation_prob=0.5,
        )

        data_loader = torch.utils.data.DataLoader(
            mixup_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                cfg["data"] + "/train_images", transform=transforms_train
            ),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
        )

    # get some random training images
    dataiter = iter(data_loader)
    images, _ = next(dataiter)

    # show images
    plt.figure(figsize=(20, 25))
    imshow(torchvision.utils.make_grid(images[0:20]))
