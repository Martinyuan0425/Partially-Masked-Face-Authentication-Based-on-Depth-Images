
"""
Created on Thu Jun 10 15:10:21 2021

@author: Yuan
"""
import os
from os import path
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
from numpy import linalg as la
import cv2
import time
import openpyxl
import tensorflow as tf
# import RGB detector
# import detect
import tools
from depth_tool import *
from utils.utils import preprocess_input, resize_image, show_config
from PIL import Image

### load face analysis module
import train_face_analysis
from model.architecture import * 
face_model = tools.initialize_model()
classifier = tf.keras.models.load_model("classifier/0707/")

### MTCNN modle
# import mtcnn
# face_detector = mtcnn.MTCNN()

# load parameter
normal_facenet, mask_facenet, face_analysis = tools.load_feature_extraction_model()
print('Load model complete!')

fps = 0
path_str = 0
input_shape = [160,160,3]
frame_3 = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
face = 0
threshold = 0
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    print("L500")
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 2 # meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

name_label = []

normal_feature = []
for files in os.listdir(r"C:\Users\606\Desktop\lab606\features\normal"):
    name_label.append(files.split('.')[0])
    normal_feature = tools.load_feature(normal_feature, normal_facenet, path = os.path.join(r"C:\Users\606\Desktop\lab606\features\normal", files))

masked_feature = []
for files in os.listdir(r"C:\Users\606\Desktop\lab606\features\mask"):
    masked_feature = tools.load_feature(masked_feature, mask_facenet, path = os.path.join(r"C:\Users\606\Desktop\lab606\features\mask", files))


# Streaming loop
try:
    while True:
        # face = 0
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        ###
        depth_frame = frames.get_depth_frame()
        ###
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
 
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        ###
        # depth_image0 = np.asanyarray(depth_frame.get_data())
        # depth_colormap0 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image0, alpha=0.03), cv2.COLORMAP_JET)
        
        
        
        ###
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        background = np.where((depth_image < clipping_distance), 0.0, 255.0)
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        
        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((gray_image, background))
        images = np.hstack((depth_colormap, color_image))
        # SVD
        restore_image, sigma = svd_compression(background, k=20)
        frame_3.insert(0, frame_3.pop())
        frame_3[0] = sigma
        
        # 連續3禎無改變偵測
        sol = try_dis_3(frame_3, False)
        upper_sigma = 0
        
        storage_img = np.zeros((640,480,3))
        
        # 特徵比對
        real_human_detector = try_dis(sigma[:4])
        
        # if sol==0 and real_human_detector:
        if sol==0:
            bg = np.where((depth_image < clipping_distance), 1, 0)
            
            ### depth image face ROI
            ROI, face, H_axis, V_axis = tools.face_detector(bg, color_image)
            
            # print(color_image.shape)
            
            ### MTCNN
            # faces, ROI, confidence = tools.detect_faces(face_detector, color_image, return_boxes = True)
            # print(boxes, confidence)
            
            ### Harr cascade
            # ROI = tools.haar_face_detector(color_image)
            
            # 判斷是否存在人臉
            if 200 > (ROI[2]-ROI[0]) > 10:  
                ### This is for MTCNN detector
                # face = faces[0]
                ### This is for MTCNN detector
                
                # Face ROI and bg-ROI position
                # upper_sigma = svd_compression(background[ROI[0]:ROI[2], ROI[1]:ROI[3]], k=10)
                cv2.rectangle(color_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0, 255, 0), 2)
                
                ### print MTCNN result
                # cv2.rectangle(color_image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)   
                
                ### face analysis model output
                label= train_face_analysis.test(face, face_model, classifier)

                # 人臉特徵比較
                face = Image.fromarray(face)
                face = resize_image(face, [input_shape[1], input_shape[0]], letterbox_image=True)
                face = np.expand_dims(preprocess_input(np.array(face, np.float32)), 0)
                
                if label == 1:
                    output = normal_facenet(face)
                    name, distance = tools.cal_distance(normal_feature, output)
                    face_analysis = "Normal"
                    threshold = 1.5
                    
                elif label == 0:
                    output = mask_facenet(face)
                    name, distance = tools.cal_distance(masked_feature, output)
                    face_analysis = "Masked"
                    threshold = 2
                    
                if distance >= threshold:
                    cv2.putText(color_image, name_label[name], (ROI[2], ROI[3]), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(color_image, face_analysis, (ROI[2], ROI[3]-20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 1, cv2.LINE_AA)
                elif distance <= threshold:
                    cv2.putText(color_image, name_label[name], (ROI[2], ROI[3]), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(color_image, face_analysis, (ROI[2], ROI[3]-20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 1, cv2.LINE_AA)
                print(name_label[name], face_analysis, distance)
        
        ### fps show
        # cv2.putText(bg_removed, str(fps), (550, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Color image', color_image)

        
        # fps = round(1/(end - start))
        
        
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window

        if  key == ord('w'):
            # plt.title("Singular value")
            # plt.xlabel("k")
            # plt.ylabel("value")
            # plt.plot(sigma, color='red')
            # plt.show()
            # plt.plot(V_axis)
            # plt.show()
            path_str += 1
            # writer = 'C:/Users/606/Desktop/'+ str(path_str)
            writer = 'C:/Users/606/Desktop/'
            # excel(sigma, writer + str(path_str) + '0630.xlsx')
            # excel(ROI, writer + '_.xlsx')
            
            cv2.imwrite(writer+str(path_str)+'.jpg', color_image[ROI[1]:ROI[3], ROI[0]:ROI[2]])

            print("saved!")
            # cv2.imwrite(writer+'_depth_image'+'.jpg', depth_image)
            # cv2.imwrite(writer+'_depth_colormap'+'.jpg', depth_colormap)
        elif  key == 27:
            cv2.destroyAllWindows()
            break
            
finally:
    pipeline.stop()
    