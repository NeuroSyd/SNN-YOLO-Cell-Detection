# SNN-YOLO Cell Detection
Cell detection is enabled with the SNN-YOLO network with neuromorphic imaging data. The current network was trained on Polystyrene-based microparticles in the size of 3, 8, 15 um, THP1 and LL/2 cell lines. 

The performance of the network was compared with conventional YOLOv3 with event-generated frames to serve as a baseline.

<img width="742" alt="Training" src="https://github.com/NeuroSyd/SNN-YOLO-Cell-Detection/assets/124959469/e01cfc9f-391e-4efc-af75-9afb6b8d4c9f">

# SNN-YOLO Cell Detection

Cell detection is enabled with the SNN-YOLO network with neuromorphic imaging data. The current network was trained on polystyrene-based microparticles in the size of 3, 8, 15 um, THP1 and LL/2 cell lines.

The performance of the network was compared with conventional YOLOv3 with event-generated frames to serve as a baseline.

<img width="742" alt="Training" src="https://github.com/NeuroSyd/SNN-YOLO-Cell-Detection/assets/124959469/e01cfc9f-391e-4efc-af75-9afb6b8d4c9f">

  
## Requirment
We do not have a sp
### Wizard
#### Torch
#### Snntorch
#### Numpy

Our environment is in the requirement.txt
The experiment platform is Ubuntu 22.04
## Data preparation:
Before using the code to transform the raw data, you need to ensure the format of the raw data. The defult data format is EVT 3. You can change the raw data format in this function in tools.py
```python
wizard = Wizard(encoding="evt3")
```
You need to test the timestamp of each event since we do not test for other format data.

After adjusting the format and setting the storage path, you can use the following commands to convert your raw data.

```python
python ./tools.py
```

## Training
After setting your own path, you can use the following command:
```python
python ./Yolo3_Train.py
```
You can adjust the number of two different kind of models by adjusting the following function:
```python
train_and_evaluate(CNN = False,train_dataset=dataset_image,test_dataset=dataset_image_t,totalNum=YOUR_NUM)
