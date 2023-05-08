'''
Katie Breen
For keypoint detection manuscript, we will take the images that had keypoints detected at 224x224 resolution and reproject back to native resolution. 
We will look up the native resolution using an image folder that as an example image of camera from the study at native resolution. 
We will then convert the detected x1,y1,x2,y2 points from the 224x224 resolution into the native resolution. 
'''

import pandas as pd
import glob 
import IPython
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nativeRes_imgs = glob.glob("/Users/catherinebreen/Documents/Chapter1/WRRsubmission/resolution_info/*")
camIDs = []
nativeRes = []

## turn into dictionary 
for img in nativeRes_imgs:
    camID = img.split('/')[-1].split('_')[0]
    image = cv2.imread(img)
    orig_h, orig_w, channel = image.shape
    camIDs.append(camID)
    nativeRes.append([orig_h, orig_w])

resDic = dict(zip(camIDs, nativeRes))

## to project 
## now get the resulting predicted x and ys 
results = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/bestModel_snowdepthcm.csv')
rows = len(results['filename'])
conversion_table = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/snowfree_table.csv')
convDic = dict(zip(conversion_table['camera'], conversion_table['conversion']))
stake_cm_dic = dict(zip(conversion_table['camera'], conversion_table['snow_free_cm']))

x1_proj, y1_proj, x2_proj, y2_proj = [], [], [], []
proj_pix_length = []
proj_cm_length = []
snow_depth = []

for i in range(0, rows):
    camID = results['Camera'][i]
    filename = results['filename'][i]
    #res = resDic[camID]
    keypoints = [results['x1_pred'][i], results['y1s_pred'][i], results['x2_pred'][i], results['y2_pred'][i]]
    keypoints = np.array(keypoints, dtype='float32')
    keypoints = keypoints.reshape(-1, 2)
    keypoints = keypoints * [orig_w / 224, orig_h / 224]
    pix_length = math.dist(keypoints[0], keypoints[1])
    cm_length = pix_length * float(convDic[camID]) ## this is pix * cm/pix conversion for each camera
    cm_depth = stake_cm_dic[camID] - float(cm_length)
    x1_proj.append(keypoints[0][0]), y1_proj.append(keypoints[0][1]), x2_proj.append(keypoints[1][0]), y2_proj.append(keypoints[1][1])
    proj_pix_length.append(pix_length)
    proj_cm_length.append(cm_length)
    snow_depth.append(cm_depth)


results['x1_proj'] = x1_proj
results['y1_proj'] = y1_proj
results['x2_proj'] = x2_proj
results['y2_proj'] = y2_proj
results['proj_pixel_length'] = proj_pix_length
results['proj_cm_length'] = proj_cm_length
results['snow_depth'] = snow_depth

results = results.sort_values(['Camera', 'filename'],
              ascending = [True, True])

#from .snow_pole_analysis import datetimeExtrac
import datetimeExtrac

## dictionary of just the SnowEx photos ##
## look up actual snow depth from the published data on SnowEx.com...
datetimeDic = datetimeExtrac.datetimeExtrac()
actual_snow_depth = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/SNEX20_SD_TLI_clean.csv')

filenames = []
snowdepths = []

for key in datetimeDic:
    camID = key.split('_')[0]
    datetime = datetimeDic[key]
    ## reformat date ##
    Date = datetime.split(' ')[0].replace(':','/')
    Time = datetime.split(' ')[1][:-3]
    DateAndTime = Date + ' ' + Time
    ## look up in actual_snow_depth table
    try: 
        sd = float(actual_snow_depth[(actual_snow_depth['Camera']==camID) & (actual_snow_depth['Date&Time']==DateAndTime)]['Snow Depth (cm)'])
        filenames.append(key), snowdepths.append(sd)
    except:
        # because it was cleaned, there will be some data not included
        filenames.append(key), snowdepths.append('na')


## now we have two dfs: 1) our results df and the actual snow depth cleaned during the NSIDC process
sd_df = pd.DataFrame({'filename':filenames, 'actual_sd':snowdepths})

## we will do a left outer join so that we keep all 972 rows from our test set, and just get a new column 
## of actual snow depths with na values for the Chewelahm Okanagan etc
results = pd.merge(results, sd_df,
                 on='filename', 
                 how='left')

## we will now export the results as csv to inspect. 
## there might be some post cleaning to do since we worked with the clean file (and not raw)
## we could try this with raw as well to see how different it is. 
results.to_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/results_with_actual_snowDepth.csv')


    # keypoints[0] = keypoints[0] * (orig_w / 224)
    # keypoints[2] = keypoints[2] * (orig_w /224)
    # keypoints[1] = keypoints[1] * (orig_h / 224)
    # keypoints[3] = keypoints[3] * (orig_h /224)

    # test the projection onto the image
    # output_keypoint = keypoints.reshape(-1, 2)
    # image = np.transpose(image, (1, 2, 0))
    # image = np.array(image, dtype='float32')
    # IPython.embed()
    # image = f'/Users/catherinebreen/Documents/Chapter1/WRRsubmission/data/native_res/{camID}/{filename}'
    # image = cv2.imread(image)
    # plt.imshow(image)
    # for p in range(output_keypoint.shape[0]):
    #     if p == 0: 
    #         plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
    #     else:
    #         plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'g.') ## bottom
    # plt.savefig('test.jpeg')
    # plt.close()
   

