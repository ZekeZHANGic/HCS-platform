import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_files = self.filter_image_files(os.listdir(image_dir))
        self.mask_files = self.filter_image_files(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = np.array(img)
        mask = np.array(mask)

        # Get object bounding boxes
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # skip background

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # all nuclei
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image = F.to_tensor(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def filter_image_files(self, files):
        image_extensions = ['.tif']
        return [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

def collate_fn(batch):
    return tuple(zip(*batch))
