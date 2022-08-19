import cv2
from mtcnn import MTCNN
import os
import time
import numpy as np
detector = MTCNN()

def auto_mask(img_path, save_path, mask_path = r"C:\Users\606\Downloads\masked-face-master\blue_mask.jpg"):
    mask_img = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w, d = image.shape
    result = detector.detect_faces(image)

    if result:
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        
        eye = int((int(keypoints['right_eye'][1])+int(keypoints['left_eye'][1]))/2)
        mid_point = int(keypoints['nose'][1] + (eye-keypoints['nose'][1])*0.8)
        img = cv2.resize(mask_img, (w, h-mid_point))
        
        mask_warped = img.astype(np.uint8)
        imask = mask_warped > 0
        # print(image.shape, imask.shape)
        
        
        for i in range(imask.shape[0]):
            for j in range(imask.shape[1]):
                if imask[i][j].any():
                    image[i+mid_point][j] = img[i][j]
                    
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        print("remove:", img_path)
        # os.remove(img_path)

def main(path, mask):
    i=0
    for files in os.listdir(path):             
        i += 1
        
        start = time.time()
        img_path = os.path.join(path, files)
        save_path = os.path.join(mask, files)
        auto_mask(img_path = img_path, save_path = save_path)

        end = time.time()
        time_c = end-start
        print(files, " saved:", i, "masked_img", 'time', time_c)
     
            
path = r"C:\Users\606\Desktop\lab606\features\normal"
mask = r"C:\Users\606\Desktop\lab606\features\mask"
main(path, mask)