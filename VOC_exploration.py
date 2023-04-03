'''

Catherine Breen

We can use deeplake package to inspect the VOC dataset, and determine the appropriate dimensions for training and testing

https://datasets.activeloop.ai/docs/ml/datasets/pascal-voc-2012-dataset/


'''

import deeplake
import IPython

IPython.embed()
ds = deeplake.load('hub://activeloop/pascal-voc-2012-train-val')

ds.images.shape
#(17125, None, None, 3) 

ds.instances['instance_mask'].shape
# (17125, None, None, None)
