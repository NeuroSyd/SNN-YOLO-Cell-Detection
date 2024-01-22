import config
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import dataset as SNN_dataset
from Model import YOLOv3,SnnYoloV3WM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from tqdm import tqdm
from utilsV3 import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    check_class_accuracy,
    plot_image,
    non_max_suppression as nms,
    create_combined_color_image,
    save_checkpoint,
    load_checkpoint,
)
import pandas as pd
import gc
import warnings
import os
from YoloV3_loss import YoloLoss
from torch.utils.data import DataLoader, random_split
import glob
torch.backends.cudnn.benchmark = True


interuptFlag = False
NeedLoad = False
TRAINSETSIZE = 0.9

Height = 720
Width = 1280
df_f = 4
roix =[int(0.60*Width)//df_f,int(0.70*Width)//df_f]
roiy = [int(0*Height)//df_f,int(1*Height)//df_f]

width = roix[1]-roix[0]
height = roiy[1] - roiy[0]

heightoffset = roiy[0]
widthoffset = roix[0]
FIRST_TIME = False
DATABASE = '../../Data/'
DATABASE_TRAIN_FOLDERNAME = 'comp'
DATABASE_TEST_FOLDERNAME = 'comp_test_v2'

warnings.filterwarnings("ignore")




def plot_couple_examples(model, loader, thresh, iou_thresh, anchors,CNN,save_path=None):
    model.eval()
    x, y,z = next(iter(loader))
    classlab = z
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, Sy, Sx, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, Sx=Sx,Sy=Sy, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = nms(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        if CNN:
            plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes,i,save_path)
        else:
            plot_image(create_combined_color_image(x[i].detach().cpu().numpy(),display=False),nms_boxes,i,classlab[i],save_path)


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y,_) in enumerate(loop):
        x = x.to(config.DEVICE).float() 
        y0, y1, y2 = (
            y[0].to(config.DEVICE).float(),
            y[1].to(config.DEVICE).float(),
            y[2].to(config.DEVICE).float(),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
    return mean_loss


def train_fnF(train_loader, model, optimizer, loss_fn, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y, _) in enumerate(loop):
        x = x.to(config.DEVICE).float()
        y0, y1, y2 = (
            y[0].to(config.DEVICE).float(),
            y[1].to(config.DEVICE).float(),
            y[2].to(config.DEVICE).float(),
        )
        
        # Forward pass
        out = model(x)
        loss = (
            loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
            + loss_fn(out[2], y2, scaled_anchors[2])
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss

def get_dataset(dataset):
    '''
    Split the dataset into train and test set
    '''
    train_size = int(TRAINSETSIZE * len(dataset))  
    test_size = len(dataset) - train_size  
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return train_dataset, test_dataset

def setup_training(CNN, train_dataset, test_dataset,rn=8):
    '''
    parameters:
        CNN: True for CNN, False for SNN
        train_dataset: training dataset
        test_dataset: test dataset
        rn: number of residual blocks numbers in SNN model
    
    '''
    global NeedLoad
    if CNN:
            model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
            checkpoint_file = "CNN_chek.pth.tar"
    else:
            model = SnnYoloV3WM(in_channels=2, numberC=config.NUM_CLASSES, time_steps=4,rn=rn).to(config.DEVICE)
            checkpoint_file = "SNN_chek.pth.tar"
            
    
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader = DataLoader(train_dataset, 8, shuffle=True)
    test_loader = DataLoader(test_dataset, 8, shuffle=False)
    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor(config.MS)).to(config.DEVICE)
    if os.path.exists(checkpoint_file) and NeedLoad:
        start_run, start_epoch ,result,finish= load_checkpoint(checkpoint_file, model, optimizer)
        if finish:
            
            start_run +=1
            start_epoch = 0
        print(f"Resuming training from run {start_run}, epoch {start_epoch}")
        NeedLoad = False
    else:
        start_run =0
        start_epoch = 0
        result = []
        NeedLoad = False

    return model, optimizer, loss_fn, train_loader, test_loader, scaler,scaled_anchors,start_run,start_epoch,result


def singleTrain(model, train_loader, test_loader, optimizer, loss_fn, scaler, scaled_anchors, num_epochs,training_result,currentepoch=0,run=0,CNN=True):
    '''
    Function for a single training and evaluation cycle
    
    '''
    training_results = training_result if currentepoch != 0 else []
    global interuptFlag
    global NeedLoad
    for epoch in range(currentepoch if currentepoch !=0 else 0, num_epochs):
        try:
            mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
            epoch_results = {"epoch": epoch, "mean_loss": mean_loss}

            # Evaluation phase every 10 epochs and update the results
            if (epoch+1) % 10 == 0 and epoch != 0:
                class_ACC, No_obj_acc, Obj_acc,time,accdata = check_class_accuracy(model, test_loader, num_classes=6 ,threshold=config.CONF_THRESHOLD)
                epoch_results.update({
                    "class_accuracy": class_ACC.item(),
                    "no_obj_accuracy": No_obj_acc.item(),
                    "obj_accuracy": Obj_acc.item(),
                    "time":time,
                    "3um":accdata[0].item(),
                    "8um":accdata[1].item(),
                    "15um":accdata[2].item(),
                    "ll2":accdata[3].item(),
                    "thp":accdata[5].item(),
                })
            if (epoch+1) % 10 == 0 and epoch != 0:
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD
                )
                mapval = mean_average_precision(
                    pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format="midpoint", num_classes=config.NUM_CLASSES
                )
                epoch_results.update({
                    "map_value": mapval.item()
                })
                if class_ACC.item() >= 95 and mapval >= 0.72:
                     if CNN:
                        save_checkpoint(model, optimizer, epoch, run,training_results,False, 
                                      f"CNN_chek_Map_{mapval}_ACC_{class_ACC.item()}.pth.tar")
                     else:
                        save_checkpoint(model,optimizer,epoch, run,training_results,False,f"SNN_chek_Map_{mapval}_ACC_{class_ACC.item()}.pth.tar")
                    
            # Save model checkpoint
            
            training_results.append(epoch_results)
            df = pd.DataFrame(training_results)
            df.to_csv('training_results.csv', index=False) # timely check
            if CNN:
                    save_checkpoint(model, optimizer, epoch, run,training_results,False, "CNN_chek.pth.tar")
            else:
                    save_checkpoint(model, optimizer, epoch, run, training_results,False,"SNN_chek.pth.tar")
            # Convert to DataFrame and save as CSV
        except KeyboardInterrupt:
            # Save the current state before exiting
            checkpoint_filename = "CNN_chek.pth.tar" if CNN else "SNN_chek.pth.tar"
            save_checkpoint(model, optimizer, epoch, run, training_results, False, checkpoint_filename)
            interuptFlag = True
            NeedLoad = True
            print("Training interrupted. Checkpoint saved.")
            break
    if CNN:
        save_checkpoint(model, optimizer, epoch, run,training_results,True, "CNN_chek.pth.tar")
    else:
        save_checkpoint(model, optimizer, epoch, run, training_results,True,"SNN_chek.pth.tar")
    df = pd.DataFrame(training_results)
    return df

def train_and_evaluate(CNN, train_dataset, test_dataset,rn=8,totalNum = 10):
    '''
    parameters:
        CNN: True for CNN, False for SNN
        train_dataset: training dataset
        test_dataset: test dataset
        rn: number of residual blocks numbers in SNN model
        totalNum: number of training runs
    '''
    all_results = []
    start_run = 1
    start_epoch = 0
    first  = True
    for run in range(totalNum):
        if interuptFlag:
            print("interrupted!")
            break
        # Initialize model, optimizer, etc. for each run
        model, optimizer, loss_fn, train_loader, test_loader,scaler, scaled_anchors,start_run,start_epoch,result = setup_training(CNN, train_dataset, test_dataset,rn=rn)
        if first == True:
            run = start_run
            current_epoch = start_epoch
            first = False
        else:
            current_epoch = 0
        print(f"Starting training run {run + 1}/{totalNum}")
        # Training and evaluation for the current run
        results = singleTrain(model, train_loader, test_loader, 
                              optimizer, loss_fn, scaler, 
                              scaled_anchors, num_epochs=config.NUM_EPOCHS,
                              currentepoch=current_epoch,run=run,CNN=CNN,training_result=result)
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(f'training_results_run_CNN_{CNN}_rn_{rn}_{run + 1}.csv', index=False)

        # Store DataFrame for each run
        all_results.append(df)
        del model, optimizer, loss_fn, train_loader, test_loader,scaler, scaled_anchors
        gc.collect()
        print(f"Finished training run {run + 1}/{totalNum}")
        print("--------------------------------")

    # Combine all DataFrames into one
    combined_results = pd.concat(all_results, keys=range(1, 11), names=['run'])
    combined_results.to_csv(f'combined_training_results_CNN_{totalNum}_rn_{rn}_{CNN}.csv')

    return combined_results

def tensor_fill(pathlist):
    for datapath in pathlist:
        npylist = glob.glob(f'{datapath}*.npy')
        filtered_npylist = [f for f in npylist if not os.path.basename(f).startswith('xypt')]
        filtered_npylist = [f for f in filtered_npylist if not os.path.basename(f).startswith('tpxy')]
        for idx,name in enumerate(filtered_npylist):
            data = np.load(name)
            temp = np.zeros((4,2,32,180),dtype=np.float32)
            mintime = min(data['t'])
            for i in range(len(data)):
                t = (data[i]['t']-mintime) // 100
                print(t)
                if t >= 4:  
                    continue
                p = data[i]['p']
                x = data[i]['x']-widthoffset
                y = data[i]['y']-heightoffset
                temp[t][p][x][y] = 1
            dirname, filename = os.path.split(name)
            new_filename = f'tpxy_filled_{filename}'
            new_filepath = os.path.join(dirname, new_filename)
            np.save(new_filepath, temp)

if __name__ == "__main__":
    
    if FIRST_TIME:
        tensor_fill([f'{DATABASE}{DATABASE_TRAIN_FOLDERNAME}/',f'{DATABASE}{DATABASE_TEST_FOLDERNAME}/'])
    
    dataset_image = SNN_dataset.FCDatasetV3P([f'{DATABASE}{DATABASE_TRAIN_FOLDERNAME}/'],roix,roiy,IDsnn=False)
    dataset_snn = SNN_dataset.FCDatasetV3P([f'{DATABASE}{DATABASE_TRAIN_FOLDERNAME}/'],roix,roiy,IDsnn=True)
    dataset_image_t = SNN_dataset.FCDatasetV3P([f'{DATABASE}{DATABASE_TEST_FOLDERNAME}/'],roix,roiy,IDsnn=False)
    dataset_snn_t = SNN_dataset.FCDatasetV3P([f'{DATABASE}{DATABASE_TEST_FOLDERNAME}/'],roix,roiy,IDsnn=True)
    #Using one dataset:
    '''
    Dataset_Train,Dataset_Test = get_dataset(dataset_snn)
    '''
    #dataset = dataset_image

    for i in [8]:
        print(f"SNN_rn{i}:")
        train_and_evaluate(CNN=False,train_dataset=dataset_snn,test_dataset=dataset_snn_t,rn=i,totalNum=5)
    print("CNN:")
    train_and_evaluate(CNN = True,train_dataset=dataset_image,test_dataset=dataset_image_t,totalNum=5)