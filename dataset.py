from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom
import cv2
import glob
import config
from albumentations.pytorch import ToTensorV2

from utilsV3 import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

def conver_label_v3(label_path,GridSizes,ClassNumber,reshape_fator_x,reshape_fator_y):
    '''
    transform txt label to tensor
    '''
    anchor = torch.tensor(config.ANCHORS[0]+config.ANCHORS[1]+config.ANCHORS[2])
    #print(anchor.shape)
    num_anchors = anchor.shape[0]
    num_anchors_per_scale = num_anchors // 3
    ignore_iou_thresh = 0.5
    bboxes = np.roll(np.loadtxt(fname=label_path,delimiter=' ',ndmin=2),4,axis=1).tolist()
    targets = [torch.zeros((num_anchors_per_scale,GridSize[1],GridSize[0],6)) for GridSize in GridSizes]
    for box in bboxes:
        iou_anchors = iou(torch.tensor(box[2:4]), anchor)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height, class_label = box
        has_anchor = [False] * 3  # each scale should have one anchor
        for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // num_anchors_per_scale
                #print(scale_idx)
                anchor_on_scale = anchor_idx % num_anchors_per_scale
                XS,YS = GridSizes[scale_idx]
                XS = XS
                YS = YS
                i, j = int(YS * y), int(XS * x)   # which cell
                #print(i,j)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    #print("yes")
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = XS * x - j,YS * y - i # both between [0,1]
                    width_cell, height_cell = (
                        width * XS,
                        height * YS,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                    #print("no")
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
    return tuple(targets)

class FCDatasetV3P(Dataset):
    '''
        pathlist: list of path to the classes
        sampling_time: total duration of the event file
        sample_bins: number of bins to sample the event file
        x: width of the sensor
        y: height of the sensor
    '''
    def __init__(
        self, pathlist=[],
        sampling_time=0.5e-6, sample_bins=100,IDsnn=True):
            super(FCDatasetV3P, self).__init__()
            self.classnum = len(pathlist)
            self.pathlist = pathlist
            self.sampling_time = sampling_time
            self.sample_bins = sample_bins
            self.data = []
            self.label = []
            self.x = int(config.IMAGE_SIZEX)
            self.y = round(config.IMAGE_SIZEY)
            self.IDsnn = IDsnn
            self.Gss = [
                 (self.x//32,self.y//32),(self.x//16,self.y//16),(self.x//8,self.y//8)
            ]
            for _ , path in enumerate(pathlist):
                if IDsnn:
                    self.eventflielist = glob.glob(f'{path}/tpxy_filled_*.npy')
                else:
                     self.eventflielist = glob.glob(f'{path}/*.png')
                    
    def __len__(self):
        return len(self.eventflielist)
    def __getitem__(self, idx):
                    eventfile  = self.eventflielist[idx]
                    dirname, filename = os.path.split(eventfile)
                    filenamelist = filename.split('.')
                    labelnamelist = filenamelist[0].split('_')
                    if self.IDsnn:
                        labelname = f'{labelnamelist[2]}_{labelnamelist[3]}.txt'
                    else:
                        labelname = f'{filenamelist[0]}.txt'
                    labelpath = os.path.join(dirname,labelname)
                    label = conver_label_v3(labelpath,GridSizes=self.Gss,ClassNumber=3,reshape_fator_x=1.777776,reshape_fator_y=2)
                    
                    if self.IDsnn:
                        event = np.load(eventfile) # t c w h
                        event = event.transpose(0,1,3,2)
                        #print(event.shape)
                        new_width = int(config.IMAGE_SIZEX ) 
                        new_height = round(config.IMAGE_SIZEY )
                        resized_event = []
                        for t in range(event.shape[0]):
                            # deal with each timestep, calculate each event coordinate
                            resized_channels = []
                            for c in range(event.shape[1]):
                                resized_channel = cv2.resize(event[t, c], (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                                resized_channels.append(resized_channel)

                            # stack all timesteps together
                            resized_timestep = np.stack(resized_channels, axis=0)  # in the time direction and keep the format [channels, height, width]
                            resized_event.append(resized_timestep)

                        resized_event = np.array(resized_event)
                        event = resized_event

                    else:
                        event = Image.open(eventfile).convert("RGB")
                        #print(event.size)
                        new_width = int(config.IMAGE_SIZEX )
                        new_height = round(config.IMAGE_SIZEY )
                        event = event.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        event = np.array(event)
                        #print(event.shape)
                        
                        if event.shape[-1] == 4:
                            event = event[:, :, :3]

                        # transpose to [channel, height, width] 
                        event = event.transpose((2, 0, 1))
                    spike = torch.tensor(event)
                    return spike, label,labelnamelist[2] if self.IDsnn else labelnamelist[0]
        