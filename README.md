# Pedestrian Detection
This repository contains code and resources for the final project of Introduction to Computer Vision (CS231 class). The objective of this research is to precisely identify pedestrians in an image, enabling a number of applications like crowd analysis, surveillance systems, and pedestrian tracking.

## Prerequisites

- Python 3.x
- PyTorch
- Torchvision
- OpenCV

## Dataset

We used the PennFudan Pedestrian dataset for training and evaluation. This dataset consists of annotated images containing pedestrians in various scenarios. Before training, we performed preprocessing steps, including resizing the images and generating bounding box annotations.
You can view all of the dataset's details by visiting this [link](https://www.cis.upenn.edu/jshi/ped_html/).

## Model Architecture
1. We employed the HOG (Histogram of Oriented Gradients) + SVM (Support Vector Machine) models for pedestrian detection.

2. We also experimented with the Faster R-CNN (Region-based Convolutional Neural Network) model for pedestrian detection as a baseline comparison to the HOG + model. This model combines a region proposal network (RPN) and a Fast R-CNN detector. It uses a backbone network, such as MobileNet or ResNet, to extract features from the input image, and then generates region proposals for potential pedestrian bounding boxes. The final detection is performed by classifying and refining these proposals.

## Installation
1. Clone the repository:
   ```shell
   git clone https://github.com/blkhanhlinh/pedestrian-detection.git

2. cd to Streamlit folder:
   ```shell
   cd pedestrian-detection
   cd Streamlit
3. Install the dependencies:
   ```shell
   pip install -r requirements.txt
4. Download the trained Faster R-CNN model:
   [Link to trained model](https://drive.google.com/file/d/1Q-YX0TtMsEd5cXXF-x9vVG_ktC91IpYi/view?usp=drive_link)
   Save the model in the '/' directory

## Performance
Our trained Faster R-CNN model achieves an average precision of 62% on the PennFudan Pedestrian test set while HOG + SVM model gains an average precision of 27%. The evaluation is based on the mean Average Precision (mAP) with an IoU threshold of 0.5.

## Acknowledgments
We would like to express our gratitude to Dr. Mai Tien Dung, our instructor, for guidance and support throughout our project. We extend our thanks to the authors of the PennFudan Pedestrian dataset for providing the annotated dataset, which was instrumental in our research. Special thanks to the open-source community for the valuable libraries and resources used in this project.

## Future Improvements
- Explore additional datasets and conduct transfer learning to improve model performance.
- Implement pedestrian tracking algorithms to track individuals across frames.

Feel free to customize and expand upon this template based on your specific project and requirements.
