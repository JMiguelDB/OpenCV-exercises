# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:25:14 2017

@author: JM
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------- ERROR measure --------------------

def meanSquareError(imagenOriginal,imagenRestaurada):
    M, N = imagenOriginal.shape[:2]
    error = np.float32(0.0)
    for x  in range(M):
        for y in range(N):
            error += ((imagenRestaurada[x][y]-imagenOriginal[x][y])**2)
    error = np.sqrt((1/(M*N))*error)
    return error
    
def histogramError(image,histogram,histogramEqualized):
    M, N = image.shape[:2]
    for x in range(256):
        error = np.abs(histogram[x] - histogramEqualized[x])
    return error/((M*N)*2)
    
# --- Generating a Gaussian and Salt and Pepper noises --------

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
      
#------ Power-law transformation ----------------      
def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
 
# ------ Main code -----------

command = "empty"
while command!="exit":
    command = input("What do you want to do?")   
    if command == "loadimage":
        command = input("What image do you want to load?")
        img = cv2.imread('../images/' + command)
        if img is None:
            print('Image not found')
        elif img.data:
            print('Image loaded correctly')
            cv2.imshow('Original image', img)
    elif img is not None:
        #---- Colour model conversion -----------
        if command == "colour":
            command = input("Choose the colour model:")
            #------- Grayscale -------
            if command == "gray":
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print('Error RGB to Gray-Scale:',meanSquareError(img,gray_image))
                command = input("Back to RGB?:")
                if command == "no":
                    img = gray_image  
                cv2.imshow('Gray image', gray_image)
            #------- CIE XYZ ---------
            elif command == "cie":
                cie_image = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
                print('Error RGB to CIE:',meanSquareError(img,cie_image))
                command = input("Back to RGB?:")
                if command == "no":
                    img = cie_image
                else:
                    rgb_image = cv2.cvtColor(cie_image, cv2.COLOR_XYZ2BGR)
                    print('Error CIE to RGB:',meanSquareError(img,rgb_image))
                    cv2.imshow('RGB image', rgb_image)
                cv2.imshow('CIE image', cie_image)
            #---------- HSV --------------
            elif command == "hsv":
                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                print('Error RGB to HSV:',meanSquareError(img,hsv_image))
                command = input("Back to RGB?:")
                if command == "no":
                    img = hsv_image
                else:
                    rgb_image = cv2.cvtColor(cie_image, cv2.COLOR_HSV2BGR_FULL)
                    print('Error HSV to RGB:',meanSquareError(img,rgb_image))
                    cv2.imshow('RGB image', rgb_image)
                cv2.imshow('HSV image', hsv_image)
            #---------- HLS ---------------
            elif command == "hls":
                hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                print('Error RGB to HLS:',meanSquareError(img,hls_image))
                command = input("Back to RGB?:")
                if command == "no":
                    img = hls_image
                else:
                    rgb_image = cv2.cvtColor(cie_image, cv2.COLOR_HLS2BGR_FULL)
                    print('Error HLS to RGB:',meanSquareError(img,rgb_image))
                    cv2.imshow('RGB image', rgb_image)
                cv2.imshow('HLS image', hls_image)
        #--------- Intensity transformation ---------
        elif command == "intensity":
            #TERMINAR 
            negative_image = 255 - img
            power_law = adjust_gamma(img, 1.5)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh3 = cv2.threshold(gray_image,150,255,cv2.THRESH_BINARY_INV)
            thresh1 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,4)
            thresh2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,3)
            M = np.ones(img.shape, dtype= "uint8") * 100
            added = cv2.add(img,M)
        #---------- Contrast enhancement with histograms -------
        elif command == "contrast":
            #TERMINAR
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
        #-------- Smoothing filter -----------  
        elif command == "smoothing":
            #TERMINAR
            command = input("Choose the smoothing filter:")
            if command == "2dconv":
                kernel = np.ones((5,5),np.float32)/25
                dst = cv2.filter2D(img,-1,kernel)
                print('Error 2D Convolution:',meanSquareError(img,dst))
                command = input("Overwrite original image?")
                if command == "yes":
                    img = dst
                cv2.imshow('2D convolution image', dst)
    else:
        print('Not image loaded yet')      
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    