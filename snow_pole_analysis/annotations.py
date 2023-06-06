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
import numpy as np 

def main():

    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Label snowpole images')
    parser.add_argument('--filepath', help='Path to image dir', default = '/Users/catherinebreen/Documents/Chapter1/other_snowpoles/CUB-L-02')
    parser.add_argument('--savedir', help='Path to save csv', default = '/Users/catherinebreen/Documents/Chapter1/other_snowpoles') 
    args = parser.parse_args()
        
    dir = glob.glob(f"{args.filepath}/*") #/*") ## path to 
    dir = sorted(dir)
    cameraNumber = args.filepath.split('/')[-1] ## get the camera number (could also use folder)

    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [],[],[],[]
    creationTimes = []
    conversions = []

    for i, file in tqdm.tqdm(enumerate(dir)): 
        if i > 1 and i <= 11: ## comment out if you want to do all the poles in the folder 
            img = cv2.imread(file)
            #plt.namedWindow('window', cv2.WINDOW_NORMAL)
            # IPython.embed()
            # plt.rcParams["figure.figsize"] = [7.00, 3.50]
            # plt.rcParams["figure.autolayout"] = True
            # plt.figure()
            # plt.plot(img[0],img[1])
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            # plt.show()
            plt.figure(figsize = (20,10))
            plt.imshow(img)
            plt.title('label top and then bottom', fontweight = "bold")
            top, bottom = plt.ginput(2)
            topX.append(top[0]), topY.append(top[1])
            bottomX.append(bottom[0]), bottomY.append(bottom[1])
            plt.close()

            PixelLength = math.dist(top,bottom)
            PixelLengths.append(PixelLength)
            
            ## to get top 10cm conversion
            conversion = 304.8/PixelLength #10/PixelLength
            print(conversion)
            conversions.append(conversion)

            ## to get snow free stake length in pixels
            print(PixelLength)

            filename.append(file.split('/')[-1])
            # IPython.embed()
            # creationTime = os.path.getctime(file)
            # dt_c = datetime.datetime.fromtimestamp(creationTime)
            # creationTimes.append(dt_c)
        else: pass

    avg_conversion = np.average(conversions)
    std_conversion = np.std(conversions)  
    df = pd.DataFrame({'filename':filename, 'topX':topX,'topY':topY, 'bottomX':bottomX, 'bottomY':bottomY, 'PixelLengths':PixelLengths, 
                       'conversions': conversions, 'mean': avg_conversion, 'std': std_conversion})
    df.to_csv(f'{args.savedir}/{cameraNumber}_fullstake_conversion.csv') # top10_conversion.csv')

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python snowpole_annotations.py --filepath '[insert filepath here]' ").
    main()