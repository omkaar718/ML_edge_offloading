from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import shutil
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annFile = '/Users/omkarp/Downloads/instances_train2017.json'
# Initialize the COCO api for instance annotations
coco=COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)

# a convenient function which can fetch a class name for a given id number.
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"
print('The class name is', getClassName(1, cats))


# Define the classes (out of the 81) which you want to see. Others will not be shown.
filterClasses = ['person']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))

#for imgId in imgIds:
for number, imgId in enumerate(imgIds):
    # load image
    if(number %1000 == 0): 
        print(number)
    img = coco.loadImgs(imgId)[0]
    '''
    I = io.imread(f"/Users/omkarp/Downloads/val2017/{img['file_name']}")/255.0
    plt.imshow(I)
    plt.axis('off')
    '''
    src = f"/Users/omkarp/Downloads/train2017/{img['file_name']}"
    dstn = f"/Users/omkarp/Documents/segmentation_dataset/train_images/{img['file_name']}"
    shutil.copyfile(src, dstn)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)
    filterClasses = ['person']
    mask = np.zeros((img['height'],img['width']))
    for i in range(len(anns)):
        mask = np.maximum(coco.annToMask(anns[i]), mask)
    #plt.imshow(mask)
    mask *= 255

    cv2.imwrite(f"/Users/omkarp/Documents/segmentation_dataset/train_masks/{img['file_name']}", mask.astype(np.uint8))

    
