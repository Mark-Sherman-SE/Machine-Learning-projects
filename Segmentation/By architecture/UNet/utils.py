import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os

from dataset import CarvanaDataset


def download_dataset(path):
    api = KaggleApi()
    api.authenticate()
    for data in ['test', 'train', 'train_masks']:
        if not os.path.isdir(path + data) or not os.listdir(path + data):
            api.competition_download_file('carvana-image-masking-challenge', data + '.zip', path=path)
            with ZipFile(path + data + '.zip', 'r') as zf:
                print("=> Extracting")
                zf.extractall(path)
            print("=> Removing zip archive")
            os.remove(path + data + '.zip')


def save_checkpoint(state_dict, filename="model_parameters.pth.tar"):
    print("=> Saving model parameters")
    torch.save(state_dict, filename)


def load_checkpoint(state_dict, model):
    print("=> Loading model parameters")
    model.load_state_dict(torch.load(state_dict))


def get_loaders(
        train_dir,
        train_maskdir,
        test_dir,
        batch_size,
        train_transform,
        test_transform,
        val_size=1600,
        num_workers=4,
        pin_memory=True
):
    dataset = CarvanaDataset(image_dir=train_dir, mask_dir=train_maskdir, train=True, transform=train_transform)
    indices = list(range(len(dataset)))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=pin_memory)
    dataset.set_transform(test_transform)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                            pin_memory=pin_memory)

    test_ds = CarvanaDataset(image_dir=test_dir, train=False, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False)

    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
        print(f"Dice score: {dice_score / len(loader)}")
        model.train()


def save_predictions_as_imgs(loader, model, batches_num=-1, folder="saved_images/", device="cuda"):
    model.eval()
    if batches_num < 1 or batches_num > len(loader):
        batches_num = len(loader)
    loader_iterator = iter(loader)
    for i in range(batches_num):
        x = next(loader_iterator).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(x, f"{folder}image_{i}.png")
        torchvision.utils.save_image(preds, f"{folder}mask_{i}.png")
    model.train()
