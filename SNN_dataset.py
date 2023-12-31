from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
import glob
# Dataset for SNN
#read in xml label
import xml.etree.ElementTree as ET

def parse_labelimg_xml(txtpath):
    annotations = []
    with open(txtpath,'r') as file:
        for line in file:
            temp = {}
            number = [float(s) for s in line.split()]
            if number[0] == int(number[0]):
                number = int(number[0])
            temp = {"labell":number[0],"cx":number[1],"cy":number[2],"w":number[3],"h":number[4]}
            annotations.append(temp)
    return annotations

def annotation_Process(annotations=[]):
    Boxlist = []
    for Box in annotations:
        label = Box["label"]
        w = Box["w"]
        h = Box["h"]
        cx = Box["cx"]
        cy = Box["cy"]
        temp = [label,cx,cy,w,h]
        Boxlist.append(temp)
    return Boxlist

#convert label to np

def conver_label(label_path,GridSize,BBoxnumber,ClassNumber):
    annotations = parse_labelimg_xml(label_path)
    annotations = annotation_Process(annotations)
    
    label = np.zeros((GridSize,GridSize,BBoxnumber*5+ClassNumber))
    for Box in annotations:

        if label[int(Box[1]*GridSize),int(Box[2]*GridSize),ClassNumber] == 0:        
            cx = Box[1]*GridSize - int(Box[1]*GridSize)
            cy = Box[2]*GridSize - int(Box[2]*GridSize)
            label[int(Box[1]*GridSize),int(Box[2]*GridSize),ClassNumber] = 1
            label[int(Box[1]*GridSize),int(Box[2]*GridSize),Box[0]] = 1
            label[int(Box[1]*GridSize),int(Box[2]*GridSize),ClassNumber+1:ClassNumber+5] = [cx,cy,Box[3],Box[4]]
    return label

     
class FCDataset(Dataset):
    '''
        pathlist: list of path to the classes
        sampling_time: total duration of the event file
        sample_bins: number of bins to sample the event file
        x: width of the sensor
        y: height of the sensor
    '''
    def __init__(
        self, pathlist=[],
        sampling_time=0.5e-6, sample_bins=100,x=128,y=128,IDsnn=True):
            super(FCDataset, self).__init__()
            self.classnum = len(pathlist)
            self.pathlist = pathlist
            self.sampling_time = sampling_time
            self.sample_bins = sample_bins
            self.data = []
            self.label = []
            self.x = x
            self.y = y
            for _ , path in enumerate(pathlist):
                if IDsnn:
                    eventflielist = glob.glob(f'{path}/tpxy_filled_*.npy')
                else:
                     eventflielist = glob.glob(f'{path}/*.png')
                
                for eventfile in eventflielist:
                    dirname, filename = os.path.split(eventfile) 
                    filenamelist = filename.split('.')
                    labelname = filenamelist[0] + '.xml'
                    labelpath = os.path.join(dirname,labelname)
                    label = conver_label(labelpath,GridSize=7,BBoxnumber=2,ClassNumber=3)
                    self.label.append(label)
                    if IDsnn:
                        event = np.load(eventfile)
                    else:
                        event = np.array(Image.open(eventfile))
                    self.data.append(event)
                    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
            event = self.data[idx]
            
            spike = torch.from_numpy(event).float()
            label = torch.from_numpy(self.label[idx]).float()
            return spike, label
        
