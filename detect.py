import tensorflow as tf
import numpy as np
import cv2
import argparse
import numpy as np
import os
from train import detect_faces, encode_face
from utils import read_image, initialize_model, face_detector
import time
import random

# def predict(image, face_model, normal_cls, mask_cls, classifier, classes, min_score=0.5):

#     faces, boxes = detect_faces(image, return_boxes=True)
#     dets = []
#     idx = 3
#     result = 0
#     if faces:
#         for face in faces:
#             encoded_face = encode_face(face, face_model)
#             encoded_face = np.expand_dims(encoded_face, axis=0)
#             input_data = tf.constant(encoded_face)
            
#             result = classifier(input_data).numpy()
#             idx = result.argmax()

#             # if idx==0:
#             #     result = mask_cls(input_data).numpy()
#             #     idx = result.argmax()
#             #     if result.max() >= min_score: 
#             #         print(f"\nA wild {classes[idx]} was found!", result.max(), '\nWearing mask')
#             #         dets.append(classes[idx])
#             #     else:
#             #         dets.append(-1)
                    
#             # elif idx==1:
#             #     result = normal_cls(input_data).numpy()
#             #     idx = result.argmax()
#             #     if result.max() >= min_score:
#             #         print(f"\nA wild {classes[idx]} was found!", result.max(), '\nNormal face')
#             #         dets.append(classes[idx])
#             #     else:
#             #         dets.append(-1)
                
#     return dets, boxes, idx

def predict(face, face_model, normal_cls, mask_cls, classifier, classes, min_score=0.7):

    dets = 'None'
    idx = 3
    result = 0
    score = float(random.uniform(0.9, 1))
    cls_label = 'normal'
    encoded_face = encode_face(face, face_model)
    encoded_face = np.expand_dims(encoded_face, axis=0)
    input_data = tf.constant(encoded_face)
            
    result = classifier(input_data).numpy()
    idx = result.argmax()
    
    if idx==0:
        result = mask_cls(input_data).numpy()
        idx_0 = result.argmax()
        if result.max() >= min_score: 
            # print(f"\nA wild {classes[idx_0]} was found!", result.max(), '\nWearing mask')
            dets = classes[idx_0]
            score = result.max()
            cls_label = 'mask'


                    
    elif idx==1:
        result = normal_cls(input_data).numpy()
        idx_1 = result.argmax()
        if result.max() >= min_score:
            # print(f"\nA wild {classes[idx_1]} was found!", result.max(), '\nNormal face')
            dets = classes[idx_1]
            score = result.max()
            cls_label = 'normal'

                
    return dets, score, cls_label

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, help="Image path to be analyzed")
    # args = parser.parse_args()

    face_model = initialize_model()
    # face_model.summary()
    labelmap_path = "labelmap.txt"
    
    # labelmap.txt
    classes = {}
    with open(labelmap_path, "r") as file:
        for i, label in enumerate(file.read().split("\n")):
            classes[i] = label
    
    # 0:mask, 1:normal, 2:sunglass
    classifier = tf.keras.models.load_model("classifier/method/")
    
    normal_cls = tf.keras.models.load_model("classifier/normal/")
    mask_cls = tf.keras.models.load_model("classifier/mask/")
    
    # if args.image_path:
    #     image = read_image(args.image_path)
    #     predict(image, face_model, classifier, classes)
    
    start = time.time()
    path = r"C:\Users\606\Desktop\val_data\0428\91color_img.jpg"
    method = ['mask', 'normal', 'sunglass']
    if path:
        image = read_image(path)
        dets, boxes, score = predict(image, face_model, normal_cls, mask_cls, classifier, classes)
        print(score, dets)
        # cv2.putText(image, str(dets[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3, cv2.LINE_AA)
        # cv2.putText(image, score, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3, cv2.LINE_AA)
        # cv2.imshow("img", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.putText(image, dets[0], (80, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(image, str(score), (80, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imwrite(r'C:\Users\606\Desktop\MASK\result_test.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cap = cv2.VideoCapture(1)

        while(True):
            # Capture frame-by-frame
            ret, img = cap.read()
            frame = img.copy()

            labels, boxes , result= predict(frame, face_model, normal_cls, mask_cls, classifier, classes)
            print(labels, boxes , result)
            for box, label in zip(boxes, labels):
                if label == -1:
                    continue
                x, y, width, height = box
                cv2.rectangle(frame, (x,y), (x+width,y+height), (255,255,255), 2)
                cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    end = time.time()
    # print(end - start)
if __name__ == "__main__":
    main()