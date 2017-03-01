# import the necessary packages
import numpy as np
import argparse
import cv2

import os
nombre = os.popen('ls fotos/').read()

def nombres(nombre):
    datos =nombre.split("\n")
    return(datos[:len(datos)-1])
nom = nombres(nombre)

#print (nom)

## este metodo se usa para recortar el codigo de barras
## el margen derecho es muy corto, ¿posibles problemas?
def recorta(box):
    x0 = box[0][0]
    x1 = box[0][0]
    y0 = box[0][0]
    y1 = box[0][0]
    for i in (range (len(box))):
        x0 = min(x0 , box[i][0])
        x1 = max(x1,  box[i][0])
        y0 = min(y0 , box[i][1])
        y1 = max(y1,  box[i][1])

    return (x0,x1,y0,y1)


def barCode(name):
    
    image = cv2.imread('../images/'+ name)
    #print (name,len(image))
    """
    if(len(image)>683):
        image = image = cv2.resize(image, (683, 384)) 
    """        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Sobel se usa para buscar los bordes de la imagen
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # al restar los valores y hacerle el valor absoulto tenemos una imagen de los filos
    # de la imagen
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # blur and threshold the image
    blurred = cv2.bilateralFilter(gradient,9,75,75)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)


    cv2.imshow("",closed)
    #cv2.waitKey()
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (_,cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
     
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the
    # image
    #cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    x0,x1,y0,y1 =recorta(box)
    #print(box)
    cropped = image[ y0:y1,x0:x1]
    """
    if ((len (cropped)>0) and (len(cropped[0]))):
        print(len(cropped),len(cropped[0]))
        _,cropped = cv2.threshold(cropped,100,255,cv2.THRESH_BINARY)    
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("cropped+"+name, cropped)
    """
    cv2.imshow("recortada", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #recorre(cropped)

"""
def decodifica():

    valores = {"00000000":1}
    #print(valores)


def recorre(image):
    a = []
    for i in range(len (image)):
        a.append( (image[45][i]))

    #print (a)

for i in range(len(nom)):
    print( i)    
    barCode(nom[i])

"""
barCode("barcode-1.jpg")



