# import third-party libraries
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from skimage.morphology import square, erosion


class MixUpCutMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder_train,
        image_folder_val=None,
        transform=None,
        mix_up_alpha=0.2,
        mix_up_prob=0.05,
        cut_mix_prob=0.05,
        beta=1.0,
        dilation_prob=0.05,
    ):
        self.dataset = datasets.ImageFolder(image_folder_train, transform=transform)
        if image_folder_val is not None:
            print("Merge val data set to trainset")
            val_dataset = datasets.ImageFolder(image_folder_val, transform=transform)
            self.dataset = torch.utils.data.ConcatDataset([self.dataset, val_dataset])
        else:
            print("Use only train")
        self.mix_up_alpha = mix_up_alpha
        self.mix_up_prob = mix_up_prob
        self.cut_mix_prob = cut_mix_prob
        self.dilation_prob = dilation_prob
        self.beta = beta
        self.dataset.classes = 250

    def mix_up(self, x1, x2, y1, y2):
        x_mix = self.mix_up_alpha * x1 + (1 - self.mix_up_alpha) * x2
        y_mix = self.mix_up_alpha * y1 + (1 - self.mix_up_alpha) * y2
        return x_mix, y_mix

    def cut_mix(self, x1, x2, y1, y2):
        lambda_ = np.random.beta(self.beta, self.beta)
        _, H, W = x1.shape
        r_x = np.random.uniform(0, W)
        r_y = np.random.uniform(0, H)
        r_w = W * np.sqrt(1 - lambda_)
        r_h = H * np.sqrt(1 - lambda_)
        x1_start, x1_end = int(max(r_x - r_w / 2, 0)), int(min(r_x + r_w / 2, W))
        y1_start, y1_end = int(max(r_y - r_h / 2, 0)), int(min(r_y + r_h / 2, H))

        x_cut = x1.clone()
        x_cut[:, y1_start:y1_end, x1_start:x1_end] = x2[
            :, y1_start:y1_end, x1_start:x1_end
        ]

        lambda_ = 1 - ((x1_end - x1_start) * (y1_end - y1_start) / (W * H))
        y_cut = lambda_ * y1 + (1 - lambda_) * y2
        return x_cut, y_cut

    def apply_dilation(self, image):
        random_size = np.random.randint(2, 3)
        binary_image = image > 0.5  # binarize

        dilated_image = np.stack(
            [erosion(channel, square(random_size)) for channel in image], axis=0
        )
        return dilated_image

    def __getitem__(self, index):
        x, y = self.dataset[index]
        num_classes = self.dataset.classes
        y_onehot = torch.zeros(num_classes).scatter_(0, torch.tensor(y).unsqueeze(0), 1)
        if np.random.rand() < self.dilation_prob:
            if torch.is_tensor(x):
                x_numpy = x.numpy()

            # Apply dilation
            x_dilated = self.apply_dilation(x_numpy)
            x = torch.from_numpy(x_dilated).to(dtype=x.dtype)

        if np.random.rand() < self.mix_up_prob:
            mix_index = np.random.randint(0, len(self.dataset))
            x_mix, y_mix = self.dataset[mix_index]
            y_mix_onehot = torch.zeros(num_classes).scatter_(
                0, torch.tensor(y_mix).unsqueeze(0), 1
            )
            x, y_onehot = self.mix_up(x, x_mix, y_onehot, y_mix_onehot)

        elif np.random.rand() < self.cut_mix_prob:
            cut_index = np.random.randint(0, len(self.dataset))
            x_cut, y_cut = self.dataset[cut_index]
            y_cut_onehot = torch.zeros(num_classes).scatter_(
                0, torch.tensor(y_cut).unsqueeze(0), 1
            )
            x, y_onehot = self.cut_mix(x, x_cut, y_onehot, y_cut_onehot)

        return x, y_onehot

    def __len__(self):
        return len(self.dataset)
