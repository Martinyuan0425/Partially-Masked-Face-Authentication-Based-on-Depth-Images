import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np


def load_sigma(path):
    S = []
    wb = load_workbook(path)
    sheet = wb['NumPy_Arr']
    for i in range(640):
        tag = 'A'+str(i+1)
        value = sheet[tag].value
        S.append(value)
    return S
path = []
for i in range (6):
    writer = 'C:/Users/606/Desktop/lab606/'+ str(i+1)
    path.append(writer +'.xlsx')
    
th = 0
c = []
a = load_sigma(path[2])
b = np.arange(0, 640, 1)
b_ = np.arange(0, 200, 1)
for i in range (200):
    c.append(a[i+20]-a[i])
    if (a[i+20]-a[i]) > 40 and th==0 and i>50:
        th = i
# c = np.array(c)
# c[c<0]=0
# plt.plot(b, a, 'b')
# plt.show()
# plt.plot(b_, c, 'r')
# plt.show()

img = cv2.imread('C:/Users/606/Desktop/lab606/3_bg_removed.jpg')
cv2.line(img, (0, th), ((640, th)), (0,255,0), 1)
cv2.imshow('bg_removed', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()