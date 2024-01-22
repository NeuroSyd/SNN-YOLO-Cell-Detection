'''
iou_width_height,intersection_over_union,non_max_suppression,mean_average_precision
are the function from : https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3
plot_image,get_evaluation_bboxes are modified to fit the format of our data
'''
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 



def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def plot_image(image, boxes, idx,classlab, save_path=None):
    """Plots predicted bounding boxes on the image and saves both the original and plotted images."""
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist

    cmap = plt.get_cmap("tab20b")
    class_labels = config.MY_LABELS if config.DATASET == 'COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    fig_orig, ax_orig = plt.subplots(1, figsize=(width / 100, height / 100))
    ax_orig.imshow(im)
    ax_orig.axis('off')

    # Save the original image
    if save_path:
        plt.savefig(f'{save_path}_original_{classlab}_{idx}.png', bbox_inches='tight', pad_inches=0,dpi=100)
        plt.close(fig_orig)
        
    # Create figure and axes for the annotated image
    fig, ax = plt.subplots(1, figsize=(width / 100, height / 100))
    ax.imshow(im)
    ax.axis('off')

    # Store class labels for the filename
    detected_classes = []

    # Create a Rectangle patch for each box
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        detected_classes.append(class_labels[class_pred])  # Add class label to list
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        text_x = upper_left_x * width
        if upper_left_y > 0.1:  # if there is space above the box
            text_y = (upper_left_y - 0.1) * height  # above the box
        else:
            text_y = (upper_left_y + box[3] + 0.1) * height  # below the box
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[class_pred],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            text_x,
            text_y,
            s=class_labels[class_pred],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred], "pad": 0, "alpha": 0.5},
        )

    # Save the annotated image
    if save_path:
        # Join all detected class labels to include in the filename
        classes_in_filename = "_".join(detected_classes)
        plt.savefig(f'{save_path}_annotated_{classes_in_filename}_{idx}.png', pad_inches=0,dpi=100, bbox_inches='tight')
        plt.close(fig)
        
def create_combined_color_image(data, colors=[(1, 1, 1), (0.5, 0.6, 1)], display=True):
    """
    Visialized the event data as a combined color image.

    parameters:
    data - event data with shape [time_steps, channels, height, width]
    colors - tuple of colors for each channel, default is [(1, 1, 1), (0.5, 0.6, 1)]
    display - show the image in a new window, default is True

    returns:
    combined_image - image with shape [height, width, 3]
    """
    #get the shape
    num_time_points, channels, height, width = data.shape

    # create a black image
    combined_image = np.zeros((height, width, 3))

    # calculate all events in each timestep and turn them into a color data.
    for t in range(num_time_points):
        for c in range(channels):
            color = colors[c]
            for i in range(3): 
                combined_image[:, :, i] += data[t, c] * color[i]

    # normalize the image
    combined_image = np.clip(combined_image / num_time_points, 0, 1)

    if display:
        # display image
        plt.imshow(combined_image)
        plt.title("Combined Color Image Over Time Steps")
        plt.axis('off')
        plt.show()

    return combined_image


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold, # threshold for class confidence score
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels,_) in enumerate(tqdm(loader)):
        x = x.to(device).float()

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            Sy = predictions[i].shape[2]
            Sx = predictions[i].shape[3]
            anchor = torch.tensor([*anchors[i]]).to(device) * torch.tensor([Sx, Sy]).unsqueeze(0).to(config.DEVICE)
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, Sx=Sx, Sy= Sy,is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(labels[2], anchor, Sx=Sx, Sy=Sy, is_preds=False)

        for idx in range(batch_size):
            #print(idx)
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, Sx,Sy, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    
    goal: to reshape the box to the original image size
    modified from: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3
        to fix the different size in the x/y direction
    
    INPUT:
    predictions: tensor of size (N, 3, Sx, Sy, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, Sx, Sy, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) #
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices_x = (
        torch.arange(Sx)
        .repeat(predictions.shape[0], num_anchors, Sy, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    cell_indices_y = (
        torch.arange(Sy)
        .repeat(predictions.shape[0], num_anchors, Sx, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # calculate the x and y coordinate of the boxes
    x = 1 / Sx * (box_predictions[..., 0:1] + cell_indices_x)
    y = 1 / Sy * (box_predictions[..., 1:2] + cell_indices_y.permute(0, 1, 3, 2, 4))
    w_h = torch.cat([
    1 / Sx * box_predictions[..., 2:3],  # 
    1 / Sy * box_predictions[..., 3:4]   # reshape the box 
        ], dim=-1)
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * Sx * Sy, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold,num_classes):
    '''
    goal: check the classificaiton accuracy
    modified: add the function to check the pridiction time and the accuracy for each class
    '''
    model.eval()
    class_correct = torch.zeros(num_classes,device=config.DEVICE)
    class_totals = torch.zeros(num_classes,device=config.DEVICE)
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y,_) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE).float()
        with torch.no_grad():
            start_time = time.time()
            out = model(x)
            end_time = time.time()
            timecom = end_time - start_time

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            class_preds = torch.argmax(out[i][..., 5:][obj], dim=-1)
            for c in range(num_classes):
                class_mask = y[i][..., 5][obj] == c
                class_correct[c] += torch.sum(class_preds[class_mask] == c)
                class_totals[c] += torch.sum(class_mask)

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
    class_accuracies = (class_correct / (class_totals + 1e-16)) * 100

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    print(f"Time is: {timecom:2f}")
    model.train()
    return (correct_class/(tot_class_preds+1e-16))*100,(correct_noobj/(tot_noobj+1e-16))*100,(correct_obj/(tot_obj+1e-16))*100,timecom,class_accuracies


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def save_checkpoint(model, optimizer, epoch, run,data, finish,accdata,filename="my_checkpoint.pth.tar"):
    torch.save({
        'run': run,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latest_result':data,
        'finish':finish,
        'class_accs':accdata,
    }, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    if ~checkpoint['finish']:
        print(checkpoint['finish'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print( checkpoint['epoch'])
        return checkpoint['run'], checkpoint['epoch'],checkpoint['latest_result'],checkpoint['finish']
    else:
        return 0,0,[],False
