import os
import cv2
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, train=False, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            _, mask = cv2.VideoCapture(mask_path).read()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype("float32")
            mask[mask == 255.0] = 1.0
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            return image, mask

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def set_transform(self, transform):
        self.transform = transform
