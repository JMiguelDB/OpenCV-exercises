# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:54:40 2017

@author: JM
"""
import cv2
import numpy as np
from barcodeDetecting import barcode

def crop(box):
    x0 = box[0][0]
    x1 = box[0][0]
    y0 = box[0][1]
    y1 = box[0][1]
    for i in (range (len(box))):
        x0 = min(x0 , box[i][0])
        x1 = max(x1,  box[i][0])
        y0 = min(y0 , box[i][1])
        y1 = max(y1,  box[i][1])

    return (x0-5,x1+5,y0,y1)

def readImage(filename):
   image = cv2.imread('../images/' + filename)
    
   if(len(image)>1200):
       image = image = cv2.resize(image, (1200, 800)) 
        
   return image

def imageProcessing(original,image):
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (_,cnts ,_) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    print(rect)
    box = np.int0(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(original, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", original)
    cv2.waitKey(0)
    
    return box,rect
    
"""
### -------------- Prueba 1: Barcode perfecto --------------------------------------------------
#Leemos el fichero
filename = "barcode.png"
image = readImage(filename)

#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,100,200)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Aplicamos la decodificacion del barcode
barcode = barcode(cropped)
barcode.read_barcode()
barcode.show()
###------------------------------------------------------------------------------------------
"""
"""
###------------- Prueba 2: Barcode en objeto ---------------------------------------------
#Leemos el fichero
filename = "barcode-3.jpg"
image = readImage(filename)

#Rotamos la imagen 180º para obtener el barcode
rot_mat = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2),180,1.0)
image = cv2.warpAffine(image, rot_mat, (image.shape[0], image.shape[1]),flags=cv2.INTER_LINEAR)
#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,50,100)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box = imageProcessing(image.copy(),thresh)

y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]


#Aplicamos la decodificacion del barcode
barcode = barcode(cropped)
barcode.read_barcode()
barcode.show()
###-------------------------------------------------------------------------------------
"""
#Leemos el fichero
filename = "girada.jpg"
image = readImage(filename)

#Rotamos la imagen 180º para obtener el barcode

#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,100,200)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box,rect = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]
rot_mat = cv2.getRotationMatrix2D((cropped.shape[0]/2, cropped.shape[1]/2),rect[2],1.0)
cropped = cv2.warpAffine(cropped, rot_mat, (cropped.shape[0]+25, cropped.shape[1]+25),flags=cv2.INTER_LINEAR)

gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#thresh = cv2.Canny(gray,70,180)
thresh = cv2.Canny(gray,70,183)
box,rect = imageProcessing(cropped.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = cropped[x1:x2,y1+15:y2]
#cv2.imshow("cropped", cropped)
#cv2.waitKey(0)

#Aplicamos la decodificacion del barcode
barcode = barcode(cropped)
barcode.read_barcode()
barcode.show()

cv2.destroyAllWindows()