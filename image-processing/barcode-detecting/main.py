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
       image = cv2.resize(image, (1200, 800)) 
        
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
box,_ = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Aplicamos la decodificacion del barcode
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  
barcode = barcode(gray)
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
box,rect = imageProcessing(image.copy(),thresh)

y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Aplicamos la decodificacion del barcode
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  
barcode = barcode(gray)
barcode.read_barcode()
barcode.show()
###-------------------------------------------------------------------------------------
"""
"""
###------------ Prueba 3: Barcode girado --------------------------------------------------
#Leemos el fichero
filename = "girada.jpg"
image = readImage(filename)

#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,100,200)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box,rect = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Giramos la imagen en funcion del grado de curvatura
rot_mat = cv2.getRotationMatrix2D((cropped.shape[0]/2, cropped.shape[1]/2),rect[2],1.0)
cropped = cv2.warpAffine(cropped, rot_mat, (cropped.shape[0]+25, cropped.shape[1]+25),flags=cv2.INTER_LINEAR)

#Calculamos de nuevo el codigo de barras para desechar las partes innecesarias de la imagen y recortamos
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,70,183)
box,rect = imageProcessing(cropped.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = cropped[x1:x2,y1+15:y2]

#Aplicamos la decodificacion del barcode
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  
barcode = barcode(gray)
barcode.read_barcode()
barcode.show()
"""
"""
### -------------- Prueba 4: Modo experto (Desencripta chino) --------------------------------------------------
#Leemos el fichero
filename = "barcode.jpeg"
image = readImage(filename)

#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,100,200)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box,_ = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Aplicamos operaciones morfologicas con un kernel con forma de linea vertical recta 
#con la que mejorar las barras de la imagen
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)

#Aplicamos la decodificacion del barcode
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  
barcode = barcode(gray)
barcode.read_barcode()
barcode.show()
###------------------------------------------------------------------------------------------
"""
"""
### -------------- Prueba 5: Lectura de Barcode con ruido --------------------------------------------------
#Leemos el fichero
filename = "barcode-saltnoise.jpg"
image = readImage(filename)

#Convertimos la imagen a gris y le aplicamos operaciones morfológicas para remover el ruido sal y pimienta
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
thresh = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Image", thresh)
cv2.waitKey(0)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow("Image", thresh)
cv2.waitKey(0)

#Aplicamos la detección de bordes de Canny a la imagen sin ruido
thresh = cv2.Canny(thresh,100,200)
cv2.imshow("Image", thresh)
cv2.waitKey(0)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box,_ = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]

#Como el barcode queda muy pequeño, lo reescalamos para mejorar la deteccion
cropped = cv2.resize(cropped, (600, 400)) 

#Aplicamos la decodificacion del barcode
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  
barcode = barcode(gray)
barcode.read_barcode()
barcode.show()
###------------------------------------------------------------------------------------------
"""
"""
### -------------- Prueba 6: Barcode con poca iluminacion --------------------------------------------------
#Leemos el fichero
filename = "barcode-oscuro.jpg"
image = readImage(filename)

#Convertimos la imagen a gris y le aplicamos la deteccion de vertices de Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(gray,100,200)

#Obtenemos el contorno del barcode y recortamos el barcode a partir del contorno
box,_ = imageProcessing(image.copy(),thresh)
y1,y2,x1,x2 = crop(box)
cropped = image[x1:x2,y1:y2]
cropped = cv2.resize(cropped, (600, 400)) 

#Se le aplican operaciones morfologicas para quitar el ruido blanco que se genera al tener poca iluminacion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
cropped = cv2.erode(cropped, kernel, iterations = 2)
cropped = cv2.dilate(cropped, kernel, iterations = 6)

#Se le aplica un threshold adaptativo que mejora la calidad en ambientes con poca iluminacion
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
cropped = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#Aplicamos la decodificacion del barcode
barcode = barcode(cropped)
barcode.read_barcode()
barcode.show()
###------------------------------------------------------------------------------------------
"""

cv2.destroyAllWindows()