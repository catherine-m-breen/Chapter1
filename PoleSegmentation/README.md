All code was adapted from https://github.com/zhmiao/GeneralSegmentation in order to segment snow poles from camera trap images for snow depth measurement. Please visit Zhongqi Miao's github page for information on model development. 

## overview
This model identifies snow depth from snow poles by classifying each pixel in the camera images of poles as either non-pole or pole pixels. The result is a segmented image of the pole. The segmentation will change depending on the snow height, which can then be converted into a snow depth by counting the number of pixels between the top and bottom of the pole and converting to cm using a conversion of pixel/cm ratio. The pixel/cm ratio is derived using an image of the pole without snow, counting the pixels that represent the pole in the image, and dividing by the full length of the pole in centimeters (Breen et al. 2022). 

## data structure information
1) original images are saved in a nested subfolder from the root folder called "JPEGImages". Each camera folder has a unique folder ID that matches the camera ID. It is critical that the first part of the image name is the camera ID followed by an "_". For example, for camera E9E, an example image is E9E_0024.JPG, where E9E corresponds to the folder and the camera ID. 
2) annotations (masked images) are saved in a subfolder from the root folder called "SegmentationClass"
3) an image list, including file names, is saved in a csv in a subfolder called "ImageSets". This is used to create the train/test splits

### Example images (image: left; mask: right)
![image](https://github.com/catherine-m-breen/Chapter1/blob/main/example_imgs/W8C_WSCT0134.JPG)
![mask](https://github.com/catherine-m-breen/Chapter1/blob/main/example_imgs/mask_W8C_WSCT0134.JPG)


## Training and evaluation
1) Before training on GPU, change the dataset root in the configuration files in `config` folder. 
    - Please also update the comet logger, and strategy function
    - Please also update the --gpus argument in the CLI to the appropriate number of GPUs
    - Increase batch size and also # of workers

2) Train: 

on local machine: 
```
python main.py --config ./configs/snowpole_plain_030923.yaml --gpus -1 --logger-type comet --session 0   
```

on single GPU: 
```
python main.py --config ./configs/snowpole_plain_030923.yaml --gpus 0 --logger_type comet --session 0 
```
Once the model is trained, a weight file will be saved to `weights` folder.

3) Evaluate:
```
python main.py --config ./configs/voc_plain_051522.yaml --gpus 0 --evaluate path_to_your_weights_file
```
**4) NOTE: Experiments will be saved to (TBD.)**

**5) NOTE: Logit adjustment method only works for binary masks now. Don't use it on VOC.**

## Basic packages:
- pytorch
- torchvision
- pytorch-lightning
- numpy
- typer (use typer[all] for rich text version)
- munch

## other resources
Breen, C. M., C. Hiemstra, C. M. Vuyovich, and M. Mason. (2022). SnowEx20 Grand Mesa Snow Depth from Snow Pole Time-Lapse Imagery, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/14EU7OLF051V. Date Accessed 04-13-2023.

