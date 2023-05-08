import IPython
import pandas as pd 
import glob
import PIL
from PIL import Image
from PIL import ExifTags

def datetimeExtrac():
    images = glob.glob('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/data/native_res/**/*.JPG')
    filenames = []
    datetimes = []

    for image in images: 
        filename = image.split('/')[-1]
        img = PIL.Image.open(image)
        exif_data = img._getexif()
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS}
        datetime = exif['DateTime']
        filenames.append(filename)
        datetimes.append(datetime)

    dictionary = dict(zip(filenames, datetimes))
    return dictionary 

#IPython.embed()