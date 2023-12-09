# import third-party libraries
import torchvision.transforms as transforms
import PIL.Image as Image


class TransformFactory:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, name):
        if name == "basic":
            # basic model
            _transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            return _transform, _transform
        elif name in ["resnet", "efficientnet", "vit"]:
            transforms_train = transforms.Compose(
                [
                    transforms.Resize(size=self.cfg["resize"]),
                    transforms.RandomAffine(
                        degrees=self.cfg["degree"],
                        translate=self.cfg["translate"],
                        scale=self.cfg["scale"],
                        shear=self.cfg["shear"],
                        fill=(255.0, 255.0, 255.0),
                        interpolation=Image.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=self.cfg["RdHflip"]),
                    transforms.ToTensor(),
                ]
            )
            transforms_val = transforms.Compose(
                [
                    transforms.Resize(size=self.cfg["resize"]),
                    transforms.ToTensor(),
                ]
            )
            return transforms_train, transforms_val
        else:
            raise ValueError("Unknown transform name")
