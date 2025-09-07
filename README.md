# Real---Time-Vehicle-License-Detection
Real-time vehicle detection system using YOLOv8 and OpenCV to identify and track vehicles in video streams. Integrates EasyOCR for license plate recognition and  sort module  for stable tracking. Designed for high-speed, frame-by-frame analysis in dynamic environments.
Pipeline Overview
This system performs real-time vehicle and license plate recognition using a multi-stage computer vision pipeline. The YOLOv8 models used for detection are pretrained and fine-tuned for inference.

1. Video Input
The system ingests video streams (live or recorded) and processes frames sequentially for object detection and tracking.

2. YOLOv8 Vehicle Detection
Each frame is passed through a pretrained YOLOv8 model to detect vehicles. Bounding boxes are generated for all detected objects classified as vehicles.

3. Vehicle Filtering
Detections are filtered based on confidence scores and class labels to retain only relevant vehicle types (e.g., cars, trucks, motorbikes).

SORT Tracking
The Simple Online and Realtime Tracking (SORT) algorithm assigns unique IDs to each vehicle and tracks them across frames using  bounding box association.

Plate ↔ Vehicle Association
License plate detections are matched to their corresponding vehicle bounding boxes based on spatial proximity and overlap logic.

7. Plate Cropping
Once associated, the license plate region is cropped from the frame for preprocessing and OCR.

8. Grayscale + Thresholding → OCR → CSV Output
The cropped plate image is converted to grayscale and thresholded to enhance text clarity. EasyOCR is then applied to extract alphanumeric characters. Recognized plate numbers, along with timestamps and vehicle IDs, are saved to a CSV file for logging and analysis.


SETUP
# Automatic-Number-Plate-Recognition-YOLOv8
## Demo






## Data

The video used in the tutorial can be downloaded [here](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing).

## Model

A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
conda create --prefix ./env python==3.10 -y
```
* Activate the environment
``` bash
source activate ./env
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run main.py with the sample video file to generate the test.csv file 
``` python
python main.py
```
* Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python add_missing_data.py
```

* Finally run the visualize.py passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
```python
python visualize.py
```


5. YOLOv8 License Plate Detection
A second pretrained YOLOv8 model is used to detect license plates within the same frame. This model is optimized for small object detection.
