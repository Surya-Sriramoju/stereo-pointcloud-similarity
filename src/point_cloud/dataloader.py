import cv2
import numpy as np
import os
import glob

class DataLoader():
    def __init__(self, data):
        """
        Dataloader class for efficient data access
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.data = data
        self.disparity_path = self.data['paths']['disparity']
        try:
            self.disparity_files = sorted(
                glob.glob(os.path.join(project_root, self.disparity_path, '*.npy'))+
                glob.glob(os.path.join(project_root, self.disparity_path, '*.npz'))
                )
        except Exception as e:
            print(f"Error loading disparity files: {e}")

        self.left_image_path = self.data['paths']['left_stereo']
        try:
            self.left_image_files = sorted(
                glob.glob(os.path.join(project_root, self.left_image_path, '*.jpeg'))+
                glob.glob(os.path.join(project_root, self.left_image_path, '*.jpg'))+
                glob.glob(os.path.join(project_root, self.left_image_path, '*.png'))
                )
        except Exception as e:
            print(f"Error loading left image files: {e}")
            
        if len(self.disparity_files) != len(self.left_image_files):
            raise ValueError("Mismatch in number of disparity and left image files")
    
    def __len__(self):
        return len(self.left_image_files)

    def __getitem__(self, idx):
        """
        Returns:  disparity, left_img
        """
        left_img = cv2.imread(self.left_image_files[idx])
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        disparity = np.load(self.disparity_files[idx]).astype(np.float32)

        if left_img is None or disparity is None:
            raise FileNotFoundError("Error Loading Data")
        return disparity, left_img