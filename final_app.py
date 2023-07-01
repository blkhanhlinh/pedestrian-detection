from matplotlib import patches
import xgboost as xgb
import streamlit as st
import torch
import joblib
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.feature import local_binary_pattern


def faster_rcnn_detection(image):
    image = Image.open(image)
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    device = torch.device('cpu')
    model.load_state_dict(torch.load('model_30.pt', map_location=device))


    model.eval()
    with torch.no_grad():
        image_tensor = T.ToTensor()(image)

        prediction = model([image_tensor.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    mask = scores > 0.8

    image_copy = np.array(image.copy())
    for box, label, score in zip(boxes[mask], labels[mask], scores[mask]):
        xmin, ymin, xmax, ymax = box.cpu().tolist()
        cv2.rectangle(image_copy, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        label_text = f"Label: {label}, Score: {score:.2f}"
        cv2.putText(image_copy, label_text, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    
    image_copy = Image.fromarray(image_copy)
    st.image(image_copy, channels="BGR")

# ===================================================

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def hog_svm_detection(image):
    model = joblib.load('models.dat')
    
    size = (64,128)
    step_size = (9,9)
    downscale = 1.25
    image = Image.open(image)
    image = np.array(image)
    image = cv2.resize(image, (300, 200))
    detections = []
    scale = 0

    for im_scaled in pyramid_gaussian(image, downscale = downscale):
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
                    
            fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)
            print(pred)
            if pred == 1:
                
                if model.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
                    int(size[0] * (downscale**scale)),
                    int(size[1] * (downscale**scale))))
        
        scale += 1
    clone = image.copy()
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.01)
    for(x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone,'Pedestrian',(x1-2,y1-2),1,0.75,(255,255,0),1)

    image_copy = Image.fromarray(clone)
    st.image(image_copy)

# ===================================================

def main():
    st.set_page_config("Pedestrian Detection")
    st.title("Pedestrian Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.header("Using Faster R-CNN")
        faster_rcnn_detection(uploaded_file)

        st.header("Using HOG combined with SVM")
        hog_svm_detection(uploaded_file)

if __name__ == "__main__":
    main()