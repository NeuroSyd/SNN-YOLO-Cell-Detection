'''
modified from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3
'''
import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "MYLabels"
UM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
IMAGE_SIZEX = 32*4
IMAGE_SIZEY = round(180*1.777776)
NUM_CLASSES = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.4
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
MS = [(IMAGE_SIZEX//32,IMAGE_SIZEY//32),(IMAGE_SIZEX//16,IMAGE_SIZEY//16),(IMAGE_SIZEX//8,IMAGE_SIZEY//8)]
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 
COCO_LABELS = [
'3um',
 '8um',
 '15um',
 'll2',
 'remain',
 'thp1',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]
MY_LABELS = [
 '3um',
 '8um',
 '15um',
 'LL2',
 'remain',
 'THP1',
]