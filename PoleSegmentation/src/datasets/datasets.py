'''
Catherine Updates: 
- updated snowpole class
- converted datatypes for cpu: https://pytorch.org/docs/stable/tensors.html
- Converted labels from 1 dim of 0 and 1s by, to 2 dim stack with inverted masks (no snow layer, and snow layer) 
- Divided by 255
- Changed the labels.long to labels.to(tensor.float32) because the loss function expected 
the actual and predicted values to match. 

'''


import os
import json
import numpy as np
from PIL import Image
#import cv2
import pandas as pd ## Catherine edit
import IPython ## Catherine edit
import matplotlib.pyplot as plt ## Catherine edit for testing 
import glob ## cat edit: for local testing 

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .ext_transforms import *
import IPython

__all__ = [
    'SNOWPOLES'
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': ExtCompose([
            ExtRandomScale((0.5, 2.0)),
            ExtRandomCrop(size=(513, 513), pad_if_needed=True),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
    'val': ExtCompose([
            ExtResize(513),
            ExtCenterCrop(513),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
}


class SNOWPOLE_DS(Dataset):
    def __init__(self,
                 rootdir,
                 dset='train',
                 transforms=None):

        self.transforms = transforms

        image_dir = os.path.join(rootdir, 'JPEGImages')

        mask_dir = os.path.join(rootdir, 'SegmentationClass')

        ####### split #########
        splits_dir = os.path.join(rootdir, 'ImageSets')
        df_data = pd.read_csv(os.path.join(splits_dir,'snowPoles_resized_labels_clean.csv')) #snowPoles_resized_labels_clean

        ## for local testing only ##
        ## check to make sure it exists in the folder of interest
        #IPython.embed()
        list_of_images = glob.glob(f'{image_dir}/*')
        list_of_images = [file.split('/')[-1] for file in list_of_images]
        df_data = df_data[df_data['filename'].isin(list_of_images)].reset_index()

        training_samples = df_data.sample(frac=0.9, random_state=100) ## same shuffle everytime

        if dset == 'train':
            file_names = training_samples['filename']
            #folder_names = [str(file.split('_')[0]) for file in file_names]
        
        if dset == 'val':
            file_names = df_data[~df_data.index.isin(training_samples.index)]['filename']
            #folder_names = [file.split('_') for file in file_names]

        ## cat edits #1 took out jpg, because my files all have .JPG extensions
        ## cat edit #2 added mask_ at beginning, because my segmented images have that naming system 
        #self.images = [os.path.join(image_dir, '{}'.format(w),'{}'.format(x)) for w, x in zip(folder_names, file_names)] 
        self.images = [os.path.join(image_dir, '{}'.format(x)) for x in file_names]
        self.masks = [os.path.join(mask_dir, 'mask_{}'.format(x)) for x in file_names]
        print(len(self.images))
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            #IPython.embed()
            img, target = self.transforms(img, target)
            ''' debugging for index error, try separating classes into two channels'''
            target1 = target
            #print('before stack', target.shape)
            target2 = (torch.div(target1,255))
            #target1, target2 = torch.tensor(target1, dtype = torch.float32), torch.tensor(target2, dtype=torch.float32)
            target_stack = torch.stack([target2, target1], dim = 0) #torch.cat([target1, target2], dim = 0) #, axis =1)
            #print('after stack', target_stack.shape)
            ''' adding a step to check for floating point type'''
            target = target_stack.to(torch.float32)#.float()
            #target = torch.tensor(target) #, dtype=torch.float32) #, device='cpu')
        
        #print('checking dtypes...', img.dtype, target.dtype)
        return img, target  #img.clone().detach(), target.clone().detach()

    def __len__(self):
        return len(self.images)

torch_version = "2.0.1"

class SNOWPOLES(pl.LightningDataModule):
    def __init__(self, conf):
        self.conf = conf
        self.prepare_data_per_node = True 
        self._log_hyperparams = False

        print("Loading data...")
        #IPython.embed()
        self.dset_tr = SNOWPOLE_DS(rootdir=self.conf.dataset_root,
                              dset='train',
                              transforms=data_transforms['train']) # inspect images without the transforms
        #IPython.embed()
        self.dset_te = SNOWPOLE_DS(rootdir=self.conf.dataset_root,
                              dset='val',
                              transforms=data_transforms['val'] )

        self.dset_te = SNOWPOLE_DS(rootdir=self.conf.dataset_root,
                              dset='val',
                              transforms=data_transforms['val'])

        print("Done.")

    def train_dataloader(self):
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )

    def val_dataloader(self): ##### cat edit: just make it all train for now
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )

    def test_dataloader(self): ##### cat edit: just make it all train for now
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )

    # def allow_zero_length_dataloader_with_multiple_devices(self):
    #     return DataLoader()