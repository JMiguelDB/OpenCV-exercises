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
    
    image = cv2.imread('images/'+ name)
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
    cv2.imwrite("manual1.jpg",closed)
	
    
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




def canny(name):
	image = cv2.imread('images/'+ name)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	thresh = cv2.Canny(gray,50,100)
	

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	
	closed = cv2.erode(closed, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)
	cv2.imwrite("canny1.jpg",closed)


def hough(name):
	print(1	)	
	img = cv2.imread('images/'+name)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,75,100,apertureSize = 3)

	lines = cv2.HoughLines(edges,1,np.pi/180,400)
	for rho,theta in lines[0]:
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0 = a*rho
	    y0 = b*rho
	    x1 = int(x0 + 1000*(-b))
	    y1 = int(y0 + 1000*(a))
	    x2 = int(x0 - 1000*(-b))
	    y2 = int(y0 - 1000*(a))

	    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

	cv2.imwrite('houghlines3.jpg',img)

def fourier(name):
	img = cv2.imread('images/'+name)
	Mat1f img = img.reshape(1);
	dft = cv2.dft((img),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	cv2.imwrite('dft_shift.jpg',img)
	rows, cols = img.shape
	crow,ccol = rows/2 , cols/2
	
	# create a mask first, center square is 1, remaining all zeros
	mask = np.zeros((rows,cols,2),np.uint8)
	mask[crow-30:crow+30, ccol-30:ccol+30] = 1
	
	# apply mask and inverse DFT
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
	cv2.imwrite('img_back.jpg',img)


fourier("barcode-3.jpg")

#hough("barcode-3.jpg")

#barCode("barcode-3.jpg")

#canny("barcode-3.jpg")

