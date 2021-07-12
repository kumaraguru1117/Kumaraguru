import cv2
import matplotlib.pyplot as plt

image=cv2.imread(r"C:\Users\omgur\Pictures\download.jfif")
print(image)
print(image.shape)
#############################################################################################################
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title("Grayed Image")
plt.imshow(gray_image)

plt.subplot(1,2,2)
plt.title("Grayed Image")
plt.imshow(image)

plt.show()
################################################################################################################
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title("Grayed Image")
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Grayed Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
################################################################################################################
import cv2

image=cv2.imread(r"C:\Users\omgur\Pictures\download.jfif")
cv2.imshow('Frame Goku', image)
cv2.waitKey(0)
###############################################################################################################
import cv2
import numpy as np

image=cv2.imread(r"C:\Users\omgur\Pictures\download.jfif")

zero=np.zeros((image.shape[0],image.shape[1]),np.uint8)
b,g,r=cv2.split(image)
print("Blue Channel:",b)
print("Green Channel:",g)
print("Red Channel:",r)


cv2.imshow('Blue',b)
##############################################################################################################
import cv2
import numpy as np

image=cv2.imread(r"C:\Users\omgur\Pictures\download.jfif")

zeros=np.zeros((image.shape[0],image.shape[1]),np.uint8)
b,g,r=cv2.split(image)
print("Blue Channel:",b)
print("Green Channel:",g)
print("Red Channel:",r)

Blue=cv2.merge([b,zeros,zeros])
Green=cv2.merge([zeros,g,zeros])
Red=cv2.merge([zeros,zeros,r])

cv2.imshow('Blue',Blue)
cv2.imshow('Green',Green)
cv2.imshow("Red",Red)
cv2.waitKey(0)
#############################################################################################################
import cv2
import numpy as np

cap = cv2.VideoCapture(r"F:\Sample1.mp4")

while cap.isOpened():
    sucess,frame=cap.read()
    if sucess:
        cv2.imshow('Frame',frame)
        k=cv2.waitKey(50)
        if k & 0xff == ord('q'):
            break
        else:
            break
            
cap.release()
cv2.destroyAllWindows()
##############################################################################################################
import cv2
import numpy as np

cap = cv2.VideoCapture(r"F:\Sample1.mp4")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucess,frame=cap.read()
    if sucess:
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray Frame',gray_frame)
        cv2.imshow('Frame',frame)
        k=cv2.waitKey(50)
        if k & 0xff == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
