from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path
from sklearn.feature_extraction import image
import numpy as np
class CustomDatasetFromImage(Dataset):
    def __init__(self, img_path, target_path, patch_size, transforms=None):
        """
        Args:
            image_path (string): path to image file
            patch_size (int): image height
            transforms: pytorch transforms for transforms and tensor conversion
        """ 
        #ToDo: Generalize for any input image(s)       
        self.img_path = Path(__file__).parent / img_path
        self.target_path = Path(__file__).parent / target_path 
        self.patch_size = patch_size
        self.transforms = transforms
        img_as_img = Image.open(self.img_path).convert('1')
        img_as_np = np.array(img_as_img)
        self.patches = image.extract_patches_2d(img_as_np, (patch_size,patch_size))
        target_as_img = Image.open(self.target_path).convert('1')
        target_as_np = np.array(target_as_img)
        self.target = target_as_np[1:149,1:111].reshape(16280) #ToDo: Generalize for any patch size


    def __getitem__(self, index):
        # must return  tensors, numbers, dicts or lists;
        patch_as_np = self.patches[index]
        patch_as_tensor = self.transforms(patch_as_np)
        label = self.target[index]
        # print(patch_as_tensor.shape)
        # torch.Size([4, 150, 112])
        return (patch_as_tensor, label)

    def __len__(self):
        return len(self.patches)  # of how many patches?


from torchvision import transforms
transformations = transforms.Compose([transforms.ToTensor()])
custom_dataset_from_image = \
    CustomDatasetFromImage('../data/J/j.png',
                           '../data/J/j_eroded.png',
                           3,
                           transformations)
# print(custom_dataset_from_image.__getitem__(0))

import torch
dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset_from_image,
                                             batch_size=32,
                                             shuffle=False)


for batch_idx, (data, target) in enumerate(dataset_loader):
    # data.shape -> torch.Size([32, 1, 28, 28])
    # target.shape -> torch.Size([32])
    print(data.shape)
    print(target.shape)
    import pdb; pdb.set_trace()
