import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

from model import VGG

torch.cuda.empty_cache()

# Code reproducibility
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Constants
BATCH_SIZE = 64
IMAGE_SIZE = (32, 32)
CHANNELS = 3
CLASSES = 10
EPOCHS = 10
LR = 1e-3

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading a dataset
    root = 'dataset/'
    train = CIFAR10(root, train=True,  transform=data_transform, download=True)
    test = CIFAR10(root, train=False, transform=data_transform, download=True)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    cifar_classes = test_loader.dataset.classes

    # Creating model
    VGG16 = VGG(in_channels=3, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], num_classes=10,
                architecture='VGG16').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(VGG16.parameters(), lr=LR, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Training
    history = []
    VGG16.train()
    for i in tqdm(range(EPOCHS)):
        running_loss = 0.0
        processed_data = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_pred = VGG16(x)

            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * x.size(0)
            processed_data += x.size(0)
        scheduler.step()
        history.append(running_loss / processed_data)

    # Save graph of train loss
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, EPOCHS + 1), history)

    plt.title('Loss by epochs')
    plt.ylabel('Cross entropy Loss')
    plt.xlabel('epochs')
    plt.xticks(range(1, EPOCHS + 1))
    plt.savefig('results/Loss by epochs.png', bbox_inches='tight')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Testing
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            y_pred = VGG16(images.to(device))
            _, predicted = torch.max(y_pred, 1)
            c = (predicted.cpu().detach() == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    with open('results/test results.txt', 'w') as test_res:
        for i in range(10):
            test_res.write('Accuracy of %5s : %2d %% \n' % (cifar_classes[i],
                  100 * class_correct[i] / class_total[i]))
