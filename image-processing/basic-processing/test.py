# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:56:30 2017

@author: JM
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def meanSquareError(imagenOriginal,imagenRestaurada):
    M, N = imagenOriginal.shape[:2]
    error = np.float32(0.0)
    for x  in range(M):
        for y in range(N):
            error += ((imagenRestaurada[x][y]-imagenOriginal[x][y])**2)
    error = np.sqrt((1/(M*N))*error)
    return error
    
def noisy(image):
      #gaussian
      aux = image.copy()
      aux = cv2.randn(aux,aux.mean(),aux.std())
      gaussian = cv2.add(image, aux)
      #salt and pepper
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return gaussian,out
   
def histogramError(image,histogram,histogramEqualized):
    M, N = image.shape[:2]
    for x in range(256):
        error = np.abs(histogram[x] - histogramEqualized[x])
    return error/((M*N)*2)
    
img = cv2.imread('../images/clahe.png',0)
img = cv2.resize(img,(600,500), interpolation = cv2.INTER_CUBIC)
gaussian,salt = noisy(img)

gaussianblur1 = cv2.GaussianBlur(img,(3,3),0)
gaussianblur2 = cv2.GaussianBlur(img,(5,5),0)
gaussianblur3 = cv2.GaussianBlur(img,(7,7),0)

median1 = cv2.medianBlur(salt,3)
median2 = cv2.medianBlur(salt,5)
median3 = cv2.medianBlur(salt,7)
"""

cv2.imshow("Original",img)
cv2.imshow("salt and pepper",salt)
cv2.waitKey(0)
cv2.imshow("salt and pepper",median1)
cv2.waitKey(0)
cv2.imshow("salt and pepper",median2)
cv2.waitKey(0)
cv2.imshow("salt and pepper",median3)
cv2.waitKey(0)
equ = cv2.equalizeHist(median1)
cv2.imshow("salt and pepper",equ)
cv2.waitKey(0)
cv2.imshow("gaussian",gaussian)
cv2.waitKey(0)
cv2.imshow("gaussian",gaussianblur1)
cv2.waitKey(0)
cv2.imshow("gaussian",gaussianblur2)
cv2.waitKey(0)
cv2.imshow("gaussian",gaussianblur3)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Error s&p 1:',meanSquareError(img,median1))
print('Error s&p 2:',meanSquareError(img,median2))
print('Error s&p 3:',meanSquareError(img,median3))
print('Error s&p 4:',meanSquareError(img,equ))

print('Error gaussian 1:',meanSquareError(img,gaussianblur1))
print('Error gaussian 2:',meanSquareError(img,gaussianblur2))
print('Error gaussian 3:',meanSquareError(img,gaussianblur3))
"""

hist1,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist1.cumsum()
cdf_normalized = cdf * hist1.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Original gray image')
plt.show()
    
equ = cv2.equalizeHist(img)
hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Original gray image with equalization')
plt.show()

print(histogramError(img,hist1,hist))