from typing import List
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import transforms


def load_pil_image(path):
    return Image.open(path).convert('RGB')


class RPDataset(Dataset):
    def __init__(self, input_dirs: List[str], image_size: int, bucket_id: int):
        self.path_pairs = []
        for input_dir in input_dirs:
            img_pathes = glob.glob(os.path.join(input_dir, '*.jpg'))
            for img_path in img_pathes:
                self.path_pairs.append((img_path, img_path))

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, index):
        path_pair = self.path_pairs[index]
        fake_path, real_path = path_pair
        fake_img = self.transform(load_pil_image(fake_path))
        real_img = self.transform(load_pil_image(real_path))
        return {
            'fake_img': fake_img,
            'gt_img': real_img
        }
