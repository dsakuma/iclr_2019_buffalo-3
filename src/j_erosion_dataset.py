from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

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
        # self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):
        # must return  tensors, numbers, dicts or lists;
        # data = # Some data read from a file or image
        img_as_img = Image.open(self.img_path)
        img_as_tensor = self.transforms(img_as_img)
        print(img_as_tensor.shape)
        # return (img, label)

    def __len__(self):
        return count  # of how many patches?


# img, label = CustomDatasetFromImage.__getitem__(None,0)


transformations = transforms.Compose([transforms.ToTensor()])
custom_dataset_from_image = \
    CustomDatasetFromImage('../data/J/j.png',
                            3,
                            transformations)
custom_dataset_from_image.__getitem__(0)

# for patches, labels in custom_dataset_from_image:
    # import pdb; pdb.set_trace(

