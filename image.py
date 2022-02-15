import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet

cfgFile = "./yolov3.cfg"
weightFile = "./yolov3.weights"
objectsFile = "./coco.names"

d = Darknet(cfgFile)
d.load_weights(weightFile)
objectName = load_class_names(objectsFile)
#get Image
iname= input("Enter Image Name: ")
myimage = cv2.imread('./{}.jpg'.format(iname))
rgbImage = cv2.cvtColor(myimage, cv2.COLOR_BGR2RGB)
changedSize = cv2.resize(rgbImage, (d.width, d.height))

iou = 0.4
nms = 0.6

square = detect_objects(d,changedSize,iou,nms)
print_objects(square, objectName)
plot_boxes(myimage, square, objectName, plot_labels=True)