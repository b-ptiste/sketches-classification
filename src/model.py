# import third-party libraries
import timm
import torch
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet50_Weights,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)


class Resnet(torch.nn.Module):
    def __init__(self, model_name, cfg, with_freeze=None):
        super(Resnet, self).__init__()
        self.model_name = model_name

        if self.model_name == "resnet18":
            model = resnet18(pretrained=True)
        if self.model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if self.model_name == "resnet101":
            model = resnet101(pretrained=True)
        if self.model_name == "resnet152":
            model = resnet152(pretrained=True)
        if self.model_name == "resnext50_32x4d":
            model = resnext50_32x4d(pretrained=True)
        if self.model_name == "resnext101_32x8d":
            model = resnext101_32x8d(pretrained=True)
        if self.model_name == "wide_resnet50_2":
            model = wide_resnet50_2(pretrained=True)
        if self.model_name == "wide_resnet101_2":
            model = wide_resnet101_2(pretrained=True)

        self.model = model

        # freeze base model
        if with_freeze is not None:
            print("\n\n Freeze Base model\n\n")
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = torch.nn.Linear(model.fc.in_features, 250)
        self.dropout1 = torch.nn.Dropout(p=cfg["drop_rate"])

    def un_freeze_model(self):
        print("\n\n UnFreeze Base model\n\n")
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)

        x = self.dropout1(x)
        x = F.log_softmax(x, dim=1)
        return x


class EfficientNet(torch.nn.Module):
    def __init__(self, model_name, cfg):
        super(EfficientNet, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=250,
            drop_rate=cfg["drop_rate"],
        )

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class Vit(torch.nn.Module):
    def __init__(self, model_name, cfg):
        super(Vit, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=250,
            drop_rate=cfg["drop_rate"],
        )

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 250)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ModelFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, model_name):
        if model_name == "basic":
            return Net()
        elif model_name == "resnet":
            return Resnet(self.cfg)
        elif model_name == "efficientnet":
            return EfficientNet(self.cfg)
        elif model_name == "vit":
            return Vit(self.cfg)
        else:
            raise ValueError(f"Unknown model: {model_name}")
