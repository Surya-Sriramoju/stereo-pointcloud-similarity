import glob
import os
import torch
import numpy as np
from PIL import Image
import yaml

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, config_path, mode="stereo", transform=None):
        with open(config_path, "r") as file:
            self.data = yaml.safe_load(file)
        self.transform = transform
        self.mode = mode
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.data_root = project_root
        self.left_path = os.path.join(project_root, self.data['paths']['left_stereo'])
        self.right_path = os.path.join(project_root, self.data['paths']['right_stereo'])
        self.left_files = sorted(
            glob.glob(os.path.join(self.left_path, '*.jpeg')) +
            glob.glob(os.path.join(self.left_path, '*.jpg')) +
            glob.glob(os.path.join(self.left_path, '*.png'))
        )
        if self.mode == "stereo":
            self.right_files = sorted(
                glob.glob(os.path.join(self.right_path, '*.jpeg')) +
                glob.glob(os.path.join(self.right_path, '*.jpg')) +
                glob.glob(os.path.join(self.right_path, '*.png'))
            )
            if len(self.left_files) != len(self.right_files):
                raise ValueError("Mismatch in left and right image count for stereo mode.")
        if len(self.left_files) == 0:
            raise ValueError(f"No images found in {self.left_path}")

    def __len__(self):
        return len(self.left_files) if self.mode == 'stereo' else len(self.left_files) - 1

    def __getitem__(self, idx):
        left_img = self.load_image(self.left_files[idx])
        if self.mode == 'stereo':
            #print("entered the loop")
            right_img = self.load_image(self.right_files[idx])
            image_pair = torch.stack([left_img, right_img], dim=0)

            
        else:
            if idx >= len(self.left_files) - 1:
                raise IndexError("Index out of range for timestep mode.")
            left_img_next = self.load_image(self.left_files[idx + 1])
            image_pair = torch.stack([left_img, left_img_next], dim=0)
        return image_pair

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img_array = np.array(img)
            img = torch.from_numpy(img_array)
            img = img.permute(2, 0, 1)
            img = img.float() / 255.0
            
        return img

        
