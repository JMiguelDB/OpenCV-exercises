# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:56:30 2017

@author: JM
"""
import cv2
import numpy as np

def meanSquareError(imagenOriginal,imagenRestaurada):
    M, N = imagenOriginal.shape[:2]
    error = np.float32(0.0)
    for x  in range(M):
        for y in range(N):
            error += (imagenRestaurada[x][y]-imagenOriginal[x][y])**2
            #print(error)
    error = np.sqrt((1/M*N)*error) 

img = cv2.imread('poc_ilum.jpg')   
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aux = image.copy()
aux= cv2.randn(aux,aux.mean(),aux.std()/100)
meanSquareError(image,aux)
cv2.imshow("Prueba",aux)
cv2.waitKey(0)
cv2.destroyAllWindows()