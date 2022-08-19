import cv2 as cv


path = r'C:\Users\606\Desktop\val_data\0428\17color_img.jpg'
img = cv.imread(path)

cv.imshow('img', img)

face_detect = cv.CascadeClassifier(r"C:\Users\606\Downloads\opencv-master\data\lbpcascades\lbpcascade_frontalface_improved.xml")

gray = cv.cvtColor(img, code = cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 3)

face_zone = face_detect.detectMultiScale(gray, scaleFactor = 1.01, minNeighbors = 5)
print("detect : \n", face_zone)

for x,y,w,h in face_zone:
    
    cv.rectangle(img, pt1=(x, y), pt2 = (x+w, y+h), color = [0,0,255], thickness = w//20)
    
    cv.circle(img, center = (x+w//2, y+h//2), radius = w//2, color = [0,255,0], thickness = w//20)
    
cv.imshow("face detection", img)

cv.waitKey(0)
cv.destroyAllWindows()

