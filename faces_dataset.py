"""Custom faces dataset."""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        #if torch.is_tensor(index):
        #    index = index.tolist()
        if index < len(self.real_image_names): #if the image is real
            label = torch.tensor([0])
            tmp_name = self.real_image_names[index]
            tmp_path = os.path.join(self.root_path,'real',tmp_name)
        else: #if the image is fake
            index = index - len(self.real_image_names)
            label = torch.tensor([1])
            tmp_name = self.fake_image_names[index]
            tmp_path = os.path.join(self.root_path, 'fake', tmp_name)
        im = Image.open(tmp_path)
        if self.transform is not None:
            im_tensor = self.transform(im)
        else:
            im_arr = np.asarray(im).transpose((2, 0, 1))
            im_tensor = torch.from_numpy(im_arr)

        return im_tensor, int(label)

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        return len(self.real_image_names)+len(self.fake_image_names)
