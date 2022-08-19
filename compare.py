import cv2
from mtcnn import MTCNN
import os
import shutil

def auto_mask():
    detector = MTCNN()
    path = r"C:\Users\606\Downloads\self-built-masked-face-recognition-dataset"
    new_path = r"C:\Users\606\Desktop\0428"
    
    count = 0
    x = 1
    
    for files in os.listdir(path):    
        for file in os.listdir(os.path.join(path, files)):
            floder_path = os.path.join(path, files, file)
            i = 0 
    
            for picture in os.listdir(floder_path):
                if(i>2):break
                img_path = os.path.join(floder_path, picture)
                check = img_path.split('.')
                if(check[-1]=='png'):continue
                orignal_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                image = orignal_image
            
                if (x==1):
                    normal_path = os.path.join((os.path.join(new_path, file)), ('normal_' + str(i) + '.jpg'))
                    cv2.imwrite(normal_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    result = detector.detect_faces(image)
                    
                    if result:    
                        bounding_box = result[0]['box']
                        keypoints = result[0]['keypoints']
                        
                        crop_img = image[bounding_box[1]:(bounding_box[1]+bounding_box[3]), bounding_box[0]:(bounding_box[0]+bounding_box[2])]
                        nose = keypoints['nose'][1] - ((keypoints['mouth_right'][1] + keypoints['mouth_left'][1])/2 - keypoints['nose'][1])*0.1
                        mouth_left = (keypoints['mouth_left'][0] - (keypoints['mouth_right'][0] - keypoints['mouth_left'][0])*0.5)
                        mouth_right = (keypoints['mouth_right'][0] + (keypoints['mouth_right'][0] - keypoints['mouth_left'][0])*0.5)
                        mouth_down = ((keypoints['mouth_right'][1] + keypoints['mouth_left'][1])/2 - keypoints['nose'][1])*0.3 + (keypoints['mouth_right'][1] + keypoints['mouth_left'][1])/2
                    
                        image[int(nose):int(mouth_down), int(mouth_left):int(mouth_right)] = [192, 192, 192]
                        
                        auto_masked_path = os.path.join((os.path.join(new_path, file)), ('auto_masked_' + str(i) + '.jpg'))
                        cv2.imwrite(auto_masked_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))                    
    
    
                if(x==2):
                    masked_path = os.path.join((os.path.join(new_path, file)), ('masked_' + str(i) + '.jpg'))
                    cv2.imwrite(masked_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                i += 1
                count += 1
        x = 2
        
    print("Numbers of pictures:", count)
    
def floder_work():
    path = r"C:\Users\606\Desktop\0428"
    for files in os.listdir(path):  
        # print(files.split('_')[-1])
        if(files.split('_')[-1]!='mask'):
            for file in os.listdir(os.path.join(path, files)):
                new_path = os.path.join(os.path.join(path,files), file)
                # print(new_path)
                check_dic = new_path.split('_')
                check_dic = check_dic[0].split('\\')
                check_dic = check_dic[-1]
                
                if(check_dic!='normal'):
                    checkpoint = new_path.split('\\')
                    new_dir = checkpoint[-2]
                    new_dir = new_dir + '_mask'
                    checkpoint[-2] = new_dir
                    # print(checkpoint)
                    save_path = None
                    for floder_dir in checkpoint:
                        if(floder_dir=='C:'):
                            save_path = floder_dir
                            save_path = save_path + '\\'
                        else:save_path = os.path.join(save_path, floder_dir)
                    shutil.move(new_path, save_path)
                
                        # print(save_path)
                # if(check_dic!='normal'):print(new_path)
            
                                           
            
        
floder_work()
    