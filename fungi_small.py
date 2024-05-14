from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imp_samp
import os
import json
import random
 

IMP_SAMP = True


n_crops_per_image = 7
imp_samp_params = {
    "patch_size": 512,
    "reduce_factor": 1,
    "scale_dog": 1,
    "grid_sep": 256,
    "map_type": "importance",
    "patches_per_image": n_crops_per_image,
    "blur_samp_map": False,
    "seed": 123
}

def important_crops_per_image(image, n_crops,imp_samp_params):
    crop_list=[]
    patcher = imp_samp.Patcher(image_path=image, **imp_samp_params)
    for i in range(n_crops):
        crop = next(patcher)
        crop_list.append(crop)    
    return crop_list


def important_random_size_crops_per_image(image, n_crops, imp_samp_params):
    crop_list = []
    for i in range(n_crops):
        # Randomly select crop size for each crop
        random_patch_size = random.randint(256, 512)
        # Update crop size parameters
        imp_samp_params['patch_size'] = random_patch_size
        imp_samp_params['grid_sep'] = int(random_patch_size/2)
        patcher = imp_samp.Patcher(image_path=image, **imp_samp_params)
        # 
        crop = next(patcher)
        crop_list.append(crop)
    return crop_list

class FungiSmall(Dataset):
    """dataset for fungi"""

    def __init__(self, images_path: list, images_class: list, transform=None, use_patches=True):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.use_patches = use_patches  # This flag controls whether to use patches or resize images

        self.patches = []
        self.labels = []
        #self.patch_to_image_map = []  # Add a new list to store the original image index corresponding to each patch
        self._generate_patches()
        #print(self.labels)

    def _generate_patches(self):
        """Generates all important patches and their labels."""
        if self.use_patches:
            for img_path, img_label in zip(self.images_path, self.images_class):
                if IMP_SAMP:
                    patches = important_crops_per_image(img_path, n_crops_per_image, imp_samp_params)
                else:
                    patches = important_random_size_crops_per_image(img_path, n_crops_per_image, imp_samp_params)
                for patch in patches:
                    self.patches.append(patch)
                    self.labels.append(img_label)  # Assuming the same label for all patches from the same image
                    #self.patch_to_image_map.append(img_idx)  # Store the original image index corresponding to the patch
        else:
             # If not using patches, simply store the image paths and their labels
            self.patches = self.images_path
            self.labels = self.images_class
            #self.patch_to_image_map = list(range(len(self.images_path)))  # Each image corresponds to its own index

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, item):
        if self.use_patches:
            patch = self.patches[item]
            label = self.labels[item]
        #print("patch,label",patch,label)

            if self.transform:
                patch = self.transform(patch)
        else:
             # If not using patches, load the full image
            image_path = self.patches[item]
            label = self.labels[item]
            patch = Image.open(image_path)  # Load the full image
            
            if self.transform:
                patch = self.transform(patch)


        return patch, label



        #print(self.images_path[item])
        #patches = important_crops_per_image(self.images_path[item],n_crops_per_image,imp_samp_params)
        #print("patches",patches)        
        
        #label = self.images_class[item]
        #print(label)

        #if self.transform:
            #for patch in patches:
                #patch = self.transform(patch)
                #return  patch, label
def read_split_data(json_file: str, root: str):
    # 
    assert os.path.exists(json_file), f"JSON file: {json_file} does not exist."
    assert os.path.exists(root), f"Root path: {root} does not exist."

    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize path and tag lists
    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []

    # Process training data
    for item in data['train']:
        full_path = os.path.join(root, item[0])  # 
        train_images_path.append(full_path)
        train_images_label.append(item[1])

    # Process test data
    for item in data['test']:
        full_path = os.path.join(root, item[0])
        test_images_path.append(full_path)
        test_images_label.append(item[1])

    #print(f"{len(train_images_path) + len(test_images_path)} images were found in the dataset.")
    #print(f"{len(train_images_path)} images for training.")
    #print(f"{len(test_images_path)} images for testing.")

    return train_images_path, train_images_label, test_images_path, test_images_label






