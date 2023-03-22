import cv2
import matplotlib.pyplot as plt 
import glob
import argparse
import tqdm
import math
import pandas as pd
import os
import datetime
import IPython

def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--filepath', help='Path to image dir', default = '/Users/catherinebreen/Documents/Chapter 1/other_snowpoles/CUB-L-02') 
    args = parser.parse_args()
        
    dir = glob.glob(f"{args.filepath}/*") ## path to 
    cameraNumber = args.filepath.split('/')[-1] ## get the camera number (could also use folder)

    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [],[],[],[]
    creationTimes = []

    for file in tqdm.tqdm(dir): 
        img = cv2.imread(file)
        plt.imshow(img)
        plt.title('label top and then bottom', fontweight = "bold")
        top, bottom = plt.ginput(2)
        topX.append(top[0]), topY.append(top[1])
        bottomX.append(bottom[0]), bottomY.append(bottom[1])
        plt.close()

        PixelLength = math.dist(top,bottom)
        PixelLengths.append(PixelLength)

        filename.append(file.split('/')[-1])
        # IPython.embed()
        # creationTime = os.path.getctime(file)
        # dt_c = datetime.datetime.fromtimestamp(creationTime)
        # creationTimes.append(dt_c)
        
    df = pd.DataFrame({'filename':filename, 'topX':topX,'topY':topY, 'bottomX':bottomX, 'bottomY':bottomY, 'PixelLengths':PixelLengths })
    df.to_csv(f'/Users/catherinebreen/Documents/Chapter 1/other_snowpoles/{cameraNumber}_validation.csv')

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()