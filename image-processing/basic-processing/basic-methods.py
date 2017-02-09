# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:35:41 2017

@author: JM
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def pintaImagen(imagen,textoGeneral,textoImagen):
    cv2.putText(imagen, textoImagen, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow(textoGeneral, imagen)
    cv2.waitKey(0)
    
def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
 
kernel = np.ones((5,5),np.float32)/25

print("¿Qué imagen quiere utilizar: Poca_iluminacion[1], Ejemplo_contraste[2], Ruido_pepper[3], Ruido_gaussiano[4]?")
tipo_img = int(input())

if tipo_img == 1:
    img = cv2.imread('../images/poc_ilum.jpg')
    print("Elegido poca iluminacion")
elif tipo_img == 2:
    img = cv2.imread('../images/clahe.png')
    print("Elegido contrast example")
elif tipo_img == 3:
    img = cv2.imread('../images/Noise_salt_and_pepper.png')
    print("Elegido pepper noise")
elif tipo_img == 4:
    img = cv2.imread('../images/gaussian_noise.jpg')
    print("Elegido gaussian noise")

print("¿Qué operaciones quiere realizar: Conversion between different colour models[1], an intensity transformation[2], a contrast enhancement operation[3], or smoothing filter[4]?")    
tipo_proc = int(input())

cv2.imshow('Imagen original', img)

if tipo_proc == 1:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cie_image = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    pintaImagen(gray_image,'Conversion a distintos modelos de color',"Escala de gris")
    pintaImagen(cie_image,'Conversion a distintos modelos de color',"Modelo CIE")
    pintaImagen(hsv_image,'Conversion a distintos modelos de color',"Modelo HSV")
    pintaImagen(hls_image,'Conversion a distintos modelos de color',"Modelo HLS")
elif tipo_proc == 2:
    negative_image = 255 - img
    power_law = adjust_gamma(img, 1.5)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh3 = cv2.threshold(gray_image,150,255,cv2.THRESH_BINARY_INV)
    thresh1 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,4)
    thresh2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,3)
    M = np.ones(img.shape, dtype= "uint8") * 100
    added = cv2.add(img,M)
    
    pintaImagen(added,'Intensity transformation','Operacion suma')
    pintaImagen(cv2.bitwise_and(gray_image,gray_image,mask = thresh3),'Intensity transformation','Threshold binario inverso')
    pintaImagen(thresh1,'Intensity transformation','Threshold adaptativo')
    pintaImagen(thresh2,'Intensity transformation','Threshold adaptativo gaussiano')
    pintaImagen(negative_image,'Intensity transformation','Negativa')
    pintaImagen(power_law,'Intensity transformation','Power law transformation')
elif tipo_proc == 3:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(gray_image.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(gray_image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title('Original gray image')
    plt.show()
    
    equ = cv2.equalizeHist(gray_image)
    hist,bins = np.histogram(equ.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(equ.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title('Original gray image with equalization')
    plt.show()

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_image)
    hist,bins = np.histogram(cl1.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(cl1.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title('Original gray image with CLAHE')
    plt.show()
    
    pintaImagen(equ,'Contrast enhancement operation','Equalization')
    pintaImagen(cl1,'Contrast enhancement operation','CLAHE')
elif tipo_proc == 4:
    dst = cv2.filter2D(img,-1,kernel)
    blur = cv2.blur(img,(5,5))
    gaussianblur = cv2.GaussianBlur(img,(5,5),0)
    median = cv2.medianBlur(img,5)
    bilateralFilter = cv2.bilateralFilter(img,9,75,75)
    
    pintaImagen(dst,'Smoothing filter','Convolution 2D')
    pintaImagen(blur,'Smoothing filter','Blur')
    pintaImagen(gaussianblur,'Smoothing filter','Gaussian blur')
    pintaImagen(median,'Smoothing filter','Salt-and-pepper noise filter')
    pintaImagen(bilateralFilter,'Smoothing filter','Filter keeping edges sharp')
    
cv2.destroyAllWindows()