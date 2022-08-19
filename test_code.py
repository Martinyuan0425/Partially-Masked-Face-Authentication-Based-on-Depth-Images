import os
import tools
import numpy as np
from utils.utils import preprocess_input, resize_image, show_config
from PIL import Image

normal_facenet, mask_facenet, face_analysis = tools.load_feature_extraction_model()

path = r"C:\Users\606\Desktop\lab606\features\normal"
feature = []
name_label = []
for files in os.listdir(path):
    name_label.append(files.split('.')[0])
    normal_feature = tools.load_feature(feature, normal_facenet, path = os.path.join(path, files))
    
def cal_distance(feature, output):
    score = []
    for f in feature:
        l1 = np.linalg.norm(output-f, axis=1)
        l1 = np.sum(np.square(output - f), axis=-1)
        score.append(l1)
    return np.argmin(score), score[np.argmin(score)]


input_shape = [160,160,3]
image_1 = Image.open(r"C:\Users\606\Desktop\val_data\0428\18color_img.jpg")
image_1 = resize_image(image_1, [input_shape[1], input_shape[0]], letterbox_image=True)
photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
output1 = normal_facenet(photo_1)

name, score = cal_distance(feature, output1)

print(name_label[name], score)