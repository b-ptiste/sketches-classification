# import standard libraries
import os

# import third-party libraries
import torch
import wandb
from torchvision import datasets

from src.data import MixUpCutMixDataset


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    with_mix_up,
    cfg,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        with_mix_up: use mixup cutmix linewidth augmentation
    """
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if with_mix_up is not None:
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(output)

            _, predicted_indices = torch.max(probabilities, 1)
            predicted_one_hot = torch.nn.functional.one_hot(
                predicted_indices, num_classes=probabilities.size(1)
            )

            # Calculate the number of correct predictions.
            correct_preds = (predicted_one_hot * target).sum(dim=1)
            num_correct = correct_preds.float().sum()

            correct += num_correct
        else:
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        with torch.no_grad():
            # to be sure nothing goes wrong with grad
            train_loss += loss

        if batch_idx % cfg["log_interval"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return train_loss, train_accuracy


def validation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)

        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    val_accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss, val_accuracy


def main(
    cfg,
    model,
    list_transforms,
    optimizer,
    scheduler=None,
    start=1,
    with_freeze=None,
    with_mix_up=None,
    path_val_in_train=None,
):
    wandb.init(
        project="RecViz Data Challenge sketch",
        name=cfg["name_experiment"],
        config=cfg,
    )
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(cfg["seed"])

    # Create experiment folder
    if not os.path.isdir(cfg["experiment"]):
        os.makedirs(cfg["experiment"])

    # load model and transform

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    if with_mix_up is not None:
        print("Using Mix Up + Cut Mix")
        # cfg['data'] + "/val_images"
        mixup_dataset = MixUpCutMixDataset(
            cfg["data"] + "/train_images",
            path_val_in_train,
            transform=list_transforms[0],
            mix_up_alpha=0.2,
            mix_up_prob=0.05,
            cut_mix_prob=0.2,
            beta=1,
            dilation_prob=0.05,
        )

        train_loader = torch.utils.data.DataLoader(
            mixup_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
        )
    else:
        print("Use only train")
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                cfg["data"] + "/train_images", transform=list_transforms[0]
            ),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
        )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(cfg["data"] + "/val_images", transform=list_transforms[1]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(start, cfg["epochs"] + start):
        if with_freeze is not None:
            if epoch == 2:
                print("\n Epoch 2 :\n")
                model.un_freeze_model()
        # training loop
        train_loss, train_accuracy = train(
            model, optimizer, train_loader, use_cuda, epoch, with_mix_up, cfg
        )
        train_metric = {
            "train/train_loss": train_loss,
            "train/train_accuracy": train_accuracy,
        }
        wandb.log(train_metric)
        # validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda)
        val_metric = {
            "val/val_loss": val_loss,
            "val/val_accuracy": val_accuracy,
        }
        wandb.log(val_metric)
        if scheduler is not None:
            # adjust_learning_rate(optimizer, epoch, 0.5, cfg['lr_warmup_epochs'], cfg['lr_warmup_method'], cfg['lr_warmup_decay'])
            scheduler.step()
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = cfg["experiment"] + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = cfg["experiment"] + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {cfg['model_name']} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
    wandb.finish()
