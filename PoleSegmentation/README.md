All code was adapted from https://github.com/zhmiao/GeneralSegmentation in order to segment snow poles from camera trap images for snow depth measurement. Please visit Zhongqi Miao's github page for information on model development. 

## data structure information
1) original images are saved in a subfolder from the root folder called "JPEGImages"
2) annotations (masked images) are saved in a subfolder from the root folder called "SegmentationClass"
3) an image list, including file names, is saved in a csv in a subfolder called "ImageSets". This is used to create the train/test splits

## Training and evaluation
1) Before training on GPU, change the dataset root in the configuration files in `config` folder. 
    - Please also update the comet logger, and accelerator function
    - Please also update the --gpus argument in the CLI to the appropriate number of GPUs
    - Increase batch size and also # of workers

2) Train: 
```
python main.py --config ./configs/snowpole_plain_030923.yaml --gpus 0,1,2,3 --logger_type comet --session 0 
```
Once the model is trained, a weight file will be saved to `weights` folder.
3) Evaluate:
```
python main.py --config ./configs/voc_plain_051522.yaml --gpus 0 --evaluate path_to_your_weights_file
```
**4) NOTE: Logit adjustment method only works for binary masks now. Don't use it on VOC.**
**5) NOTE: Experiments will be saved to (TBD.)

## Basic packages:
- pytorch
- torchvision
- pytorch-lightning
- numpy
- typer
- munch