from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path
from sklearn.feature_extraction import image
import numpy as np
class CustomDatasetFromImage(Dataset):
    def __init__(self, img_path, patch_size, transforms=None):
        """
        Args:
            image_path (string): path to image file
            patch_size (int): image height
            transforms: pytorch transforms for transforms and tensor conversion
        """        
        self.img_path = Path(__file__).parent / img_path
        self.patch_size = patch_size
        self.transforms = transforms
        img_as_img = Image.open(self.img_path).convert('1')
        img_as_np = np.array(img_as_img)
        self.patches = image.extract_patches_2d(img_as_np, (patch_size,patch_size))
        print(self.patches.shape)


    def __getitem__(self, index):
        # must return  tensors, numbers, dicts or lists;
        patch_as_np = self.patches[index]
        patch_as_tensor = self.transforms(patch_as_np)
        print(patch_as_tensor.shape)
        return patch_as_tensor
        # print(self.patches.shape)
        # torch.Size([4, 150, 112])
        # return (img, label)

    def __len__(self):
        return len(self.patches)  # of how many patches?


# img, label = CustomDatasetFromImage.__getitem__(None,0)

from torchvision import transforms
transformations = transforms.Compose([transforms.ToTensor()])
custom_dataset_from_image = \
    CustomDatasetFromImage('../data/J/j.png',
                            5,
                            transformations)
print(custom_dataset_from_image.__getitem__(0))
# print(custom_dataset_from_image.__len__())

# for patches, labels in custom_dataset_from_image:
    # import pdb; pdb.set_trace(

