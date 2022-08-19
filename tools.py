from model.architecture import *
import tensorflow as tf
import numpy as np
from PIL import Image
from utils.utils import preprocess_input, resize_image, show_config
import cv2
import os
import matplotlib.pyplot as plt


def load_feature(feature_label, normal_facenet, path):
    input_shape = [160,160,3]
    image_1 = Image.open(path)
    image_1 = resize_image(image_1, [input_shape[1], input_shape[0]], letterbox_image=True)
    photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
    output1 = normal_facenet(photo_1)
    feature_label.append(output1)
    return feature_label

def read_image(image_path:str):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def initialize_model():
    face_encoder = InceptionResNetV2()
    path = "model/facenet_keras_weights.h5"
    
    # face_encoder = create_inception_v4()
    # path = "model/inception_v4.h5"
    
    face_encoder.load_weights(path)
    return face_encoder

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

# def detect_parameter():
#     face_model = initialize_model()
#     # face_model.summary()
#     labelmap_path = "labelmap.txt"
#     classes = {}
#     normal_cls = tf.keras.models.load_model("classifier/normal/")
#     mask_cls = tf.keras.models.load_model("classifier/mask/")
#     classifier = tf.keras.models.load_model("classifier/method/")
    
#     with open(labelmap_path, "r") as file:
#         for i, label in enumerate(file.read().split("\n")):
#             classes[i] = label
#     return face_model, normal_cls, mask_cls, classifier, classes

def load_feature_extraction_model():
    from nets.facenet import facenet
    from nets.facenet import face_analysis_model
    normal_model_path = r"C:\Users\606\Desktop\lab606\logs\ep001-loss0.773-val_loss0.576.h5"
    normal_facenet = facenet(input_shape = [160, 160, 3], backbone = "inception_resnetv2", mode = "predict")
    normal_facenet.load_weights(normal_model_path, by_name=True, skip_mismatch=True)
    
    mask_model_path = r"C:\Users\606\Desktop\lab606\logs\facenet_mask\ep030-loss0.019-val_loss0.015.h5"
    mask_facenet = facenet(input_shape = [160, 160, 3], backbone = "inception_resnetv2", mode = "predict")
    mask_facenet.load_weights(mask_model_path, by_name=True, skip_mismatch=True)
    
    face_analysis_path = r"C:\Users\606\Desktop\lab606\logs\face_analyze\ep020-loss0.214-val_loss0.214.h5"
    face_analysis = face_analysis_model(input_shape = [160, 160, 3], num_classes = 2, backbone = "mobilenet", mode = "train")
    face_analysis.load_weights(face_analysis_path)
    
    # face_analysis.summary()
    # normal_facenet.summary()
    # mask_facenet.summary()
    
    return normal_facenet, mask_facenet, face_analysis


def face_detector(bg, img):
    ROI = [0,0,0,0]
    
    #縱向掃描(bg)
    V_axis = np.sum(bg, axis=1)
    for i in range(455):
        diff = V_axis[i+25] - V_axis[i]
        if V_axis[i] > 60 and ROI[1]==0:
            ROI[1] = i
            i += 20
        if ROI[1]!=0 and diff>75:
            ROI[3] = i
            break
    crop_bg = bg[ROI[1]:ROI[3], 0:640]
    height = int((ROI[3] - ROI[1])*0.5)
    
    #橫向掃描(crop_bg)w
    H_axis = np.sum(crop_bg, axis=0)
    for j in range(640):
        if H_axis[j] > height:
            if ROI[0] == 0:
                ROI[0] = j  
                ROI[2] = ROI[0]
            ROI[2] += 1
    
    face = img[ROI[1]:ROI[3], ROI[0]:ROI[2]]
    
    
    ROI[0] = ROI[0] + int((ROI[2]-ROI[0]) * 0.2)
    # ROI : [x_min, y_min, x_max, y_max] 
    return ROI, face, H_axis, V_axis

def detect_faces(face_detector, img, min_score=0.9, return_boxes=False):
    # face_detector = mtcnn.MTCNN()

    detections = face_detector.detect_faces(img)
    faces = []
    # boxes = []
    boxes = [0,0,0,0]
    confidence = 0
    for detection in detections:
        if detection["confidence"] >= min_score:
            x, y, width, height = detection['box']
            xmax, ymax = x+width , y+height
            faces.append(img[y:ymax, x:xmax] )
            # boxes.append((x, y, xmax, ymax))
            
            boxes[0] = x
            boxes[1] = y
            boxes[2] = xmax
            boxes[3] = ymax
            
            confidence = detection["confidence"]
    if return_boxes:
        return faces, boxes, confidence

    return 0

def cal_distance(feature, output):
    score = []
    for f in feature:
        l1 = np.linalg.norm(output-f, axis=1)
        l1 = np.sum(np.square(output - f), axis=-1)
        score.append(l1)
    return np.argmin(score), score[np.argmin(score)]
    
def haar_face_detector(img):
    face_cascade = cv2.CascadeClassifier(r"C:\Users\606\Desktop\lab606\data\haarcascade_frontalface_default.xml")
    
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)
    
    if face_rect != ():
        return face_rect[0]


normal_facenet, mask_facenet, face_analysis = load_feature_extraction_model()
name_label = []
normal_feature = []
for files in os.listdir(r"C:\Users\606\Desktop\lab606\features\normal"):
    name_label.append(files.split('.')[0])
    normal_feature = load_feature(normal_feature, normal_facenet, path = os.path.join(r"C:\Users\606\Desktop\lab606\features\normal", files))

# print(normal_feature[0][0])
plt.title(name_label[2])
plt.plot(normal_feature[2][0], color = "green")
plt.show()