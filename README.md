# Video Frame Annotation Using YOLOv5

## Project Overview
This Python project automates the process of video frame annotation using the YOLOv5 model, pre-trained on the COCO dataset. It extracts frames from a specified video file, uses YOLOv5 to detect and label objects in each frame, and saves these annotated frames to disk. This tool is invaluable for developing training datasets for machine learning models, especially in computer vision.

## Features
- Automatic Object Detection**: Leverages YOLOv5 for detecting objects in video frames.
- Frame-by-Frame Annotation**: Annotations include object labels and are saved with each frame.
- Output Storage**: Annotated frames and a CSV file containing labels for each frame are saved.

## Dependencies
- OpenCV
- Pandas
- PyTorch
- Torchvision (for YOLOv5)


##Output
- Annotated images will be saved in the annotated_images directory.
- A CSV file named annotations.csv, containing frame numbers and corresponding labels, will be generated in the project directory.

