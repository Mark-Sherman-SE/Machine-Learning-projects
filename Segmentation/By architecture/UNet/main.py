import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import UNet
from utils import *

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
VAL_SIZE = 1600  # Each car was photographed 16 times
BATCHES_TO_SAVE = 5
PIN_MEMORY = True
LOAD_MODEL = False
CHECKPOINT_PATH = "model_parameters.pth.tar"
DOWNLOAD_FILES = False
DOWNLOAD_PATH = "dataset/"
TRAIN_IMG_DIR = DOWNLOAD_PATH + "train/"
TRAIN_MASK_DIR = DOWNLOAD_PATH + "train_masks/"
TEST_IMG_DIR = DOWNLOAD_PATH + "test/"
RESULTS_PATH = "results/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    if DOWNLOAD_FILES:
        download_dataset(DOWNLOAD_PATH)

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        VAL_SIZE,
        NUM_WORKERS,
        PIN_MEMORY
    )

    ii = next(iter(test_loader))
    print(type(ii))
    print(type(ii[0]))
    print(ii)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_PATH, model)
        check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

    save_checkpoint(model.state_dict())

    check_accuracy(val_loader, model, device=DEVICE)

    save_predictions_as_imgs(
        test_loader, model, batches_num=BATCHES_TO_SAVE, folder=RESULTS_PATH, device=DEVICE
    )


if __name__ == '__main__':
    main()
