from keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing import image
import os
from PIL import Image
import random

train_cats_dir = r'C:\Users\606\Desktop\0512test\yang\42color_img.jpg'

def augumentation_test():
    datagen = ImageDataGenerator(
        rotation_range=40, # 角度值，0~180，影象旋轉
        width_shift_range=0.2, # 水平平移，相對總寬度的比例
        height_shift_range=0.2, # 垂直平移，相對總高度的比例
        shear_range=0.2, # 隨機錯切換角度
        zoom_range=0.2, # 隨機縮放範圍
        horizontal_flip=True, # 一半影象水平翻轉
        fill_mode='nearest' # 填充新建立畫素的方法
    )
    
    from keras.preprocessing import image # 影象預處理模組
    
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    print(fnames)
    img_path = fnames[0] # 選擇一張圖片進行增強
    img = image.load_img(img_path) # 讀取影象並調整大小
    x = image.img_to_array(img) # 形狀轉換為(150,150,3)的Numpy陣列
    x = x.reshape((1,) + x.shape)
    
    i = 0
    # 生成隨機變換後圖像批量，迴圈是無限生成，也需要我們手動指定終止條件
    for batch in datagen.flow(x, batch_size=1):
        # plt.figure(0)
        print(batch[0].shape)
        image.array_to_img(batch[0])
        cv2.imshow('img', img)
        cv2.waitKey()
        # imgplot = plt.show(image.array_to_img(batch[0]))
        # plt.savefig(r'C:\Users\606\Desktop\0512test\yang\aug.jpg')
        break
    cv2.destroyAllWindows()
    
def PIL_rotate(image, save_path, angle):
    img = Image.open(image)
    # imgR = img.transpose(Image.ROTATE_180)
    imgR = img.rotate(angle, Image.BILINEAR )
    imgR.save(save_path)
    
# save_path = r'C:\Users\606\Desktop\0512test\yang\aug.jpg'
# PIL_rotate(train_cats_dir, save_path)


path = r'C:\Users\606\Desktop\0713\YuChen'


def main(path, num):
    count = 0
    for files in os.listdir(path):
        for i in range (10):
            original_image = os.path.join(path, files)
            save_path = os.path.join(path, '_' + str(count) + '.jpg')
            angle = random.randint(-10,10)
            # print(original_image, save_path, angle)
            PIL_rotate(original_image, save_path, angle)
            count += 1
    print(count, " images created.")
    
main(path, 5)





# path = r'C:\Users\606\Desktop\mask-20220524\mask'
# numbers = 0
# for files in os.listdir(path):
#     count = 0
#     for file in os.listdir(os.path.join(path, files)):
#         check = file.split('.')[-1]
#         if check != 'jpg':
#             print(os.path.join(os.path.join(path, files), file))
            # os.remove(os.path.join(os.path.join(path, files), file))
        # count += 1
        # original_image = os.path.join(os.path.join(path, files), file)
        # img = Image.open(original_image)
        # save_path = r'C:\Users\606\Desktop\2_label_data\mask' + '\\' + str(numbers) + '.jpg'
        # if count > 1: continue
        # numbers += 1
        # img.save(save_path)
# print(numbers)
