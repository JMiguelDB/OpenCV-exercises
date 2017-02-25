# -*- coding: utf-8 -*-

import numpy as np
import cv2


def align_boundary(img, cursor, begin, end):
    if (img[cursor[1]][cursor[0]] == end):
        while (img[cursor[1]][cursor[0]] == end):
            cursor[0] += 1
    else:
        while (img[cursor[1]][cursor[0]-1] == begin):
            cursor[0] -= 1;
      
def read_digit(img, cursor, unit_width, Lcode, Gcode,Rcode, position, SPACE, BAR):
  # Read the 7 consecutive bits.
  pattern = [0, 0, 0, 0, 0, 0, 0]
  for i in range(len(pattern)):
    for j in range(unit_width):
      if (img[cursor[1]][cursor[0]] == BAR):
        pattern[i] += 1
      cursor[0] += 1

    if (pattern[i] == 1 and img[cursor[1]][cursor[0]] == BAR 
        or pattern[i] == unit_width-1 and img[cursor[1]][cursor[0]] == SPACE):
      cursor[0] -= 1
 
  # Convert to binary, consider that a bit is set if the number of
  # bars encountered is greater than a threshold.
  threshold = unit_width / 2
  v = ""
  for i in range(len(pattern)):
    if pattern[i] >= threshold:
        v += str(1)
    else:
        v += str(0)
  
  encoding=""
  # Lookup digit value.
  if (position == "LEFT"):
    digit = Lcode.get(v)
    encoding = "L"
    if digit is None:
       digit = Gcode.get(v)
       encoding = "G"
    align_boundary(img, cursor, SPACE, BAR)
  else:
    digit = Rcode.get(v)
    align_boundary(img, cursor, BAR, SPACE) 
  print("El codigo vale", v, "Digito", digit)
  return digit,encoding
  
def skip_quiet_zone(img, cursor, SPACE):
  while (img[cursor[1]][cursor[0]] == SPACE):
      cursor[0]+=1
  
def read_lguard(img, cursor, BAR, SPACE):
  widths = [ 0, 0, 0 ]
  pattern = [ BAR, SPACE, BAR ]
  for i in range(len(widths)):
    while (img[cursor[1]][cursor[0]] == pattern[i]):
      cursor[0]+=1
      widths[i]+=1
  return widths[0];

def skip_mguard(img, cursor, BAR, SPACE):
  pattern = [ SPACE, BAR, SPACE, BAR, SPACE ]
  for i in range(len(pattern)):
    while (img[cursor[1]][cursor[0]] == pattern[i]):
      cursor[0]+=1

def read_barcode(filename):
  #Definimos todas las variables necesarias 
  img = cv2.imread('../images/' + filename, 0)
  img = cv2.bitwise_not(img)
  _,img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
  cv2.imshow("original",img)
  #img = cv2.threshold(img, 128,255, cv2.THRESH_BINARY)
  
  #Definimos una lista como el cursor con los ejes [X,Y]
  cursor = [0,int(len(img[0]) / 2)]

  #Se definen las dos regiones que se encuentran dentro del codigo de barras
  Lcode = {"0001101":0,"0011001":1,"0010011":2,"0111101":3,"0100011":4,
             "0110001":5,"0101111":6,"0111011":7,"0110111":8,"0001011":9}
  
  Gcode = {"0100111":0,"0110011":1,"0011011":2,"0100001":3,"0011101":4,
             "0111001":5,"0000101":6,"0010001":7,"0001001":8,"0010111":9}          
             
  Rcode = {"1110010":0,"1100110":1,"1101100":2,"1000010":3,"1011100":4,
             "1001110":5,"1010000":6,"1000100":7,"1001000":8,"1110100":9}
             
  typeEncoding = {"LLLLLL":0,"LLGLGG":1,"LLGGLG":2,"LLGGGL":3,"LGLLGG":4,
             "LGGLLG":5,"LGGGLL":6,"LGLGLG":7,"LGLGGL":8,"LGGLGL":9}
  #Definimos los espacios en blanco y las barras negras
  SPACE = 0
  BAR = 255

  ## --------- Aplicado por Luis --------
  #cv::bitwise_not(img, img);
  #cv::threshold(img, img, 128, 255, cv::THRESH_BINARY);
  ## ------------------------------
  #Nos saltamos la primera zona de seguridad
  skip_quiet_zone(img,cursor, SPACE)
  #Calibramos el valor para que se coloque en la primera barra del codigo izquierdo
  unit_width = read_lguard(img, cursor, BAR, SPACE)
  #Almacenamos los digitos asociados al valor del codigo izquierdo
  #std::vector<int> digits;
  digits = ""
  encoding = ""
  for i in range(6):
    d,e = read_digit(img, cursor, unit_width, Lcode, Gcode, Rcode, "LEFT", SPACE, BAR)   
    digits += str(d)
    encoding += e
  
  #Saltamos las barras de seguridad del medio
  skip_mguard(img, cursor, BAR, SPACE)
 
  #Almacenamos los digitos asociados al valor del codigo derecho
  for i in range(6):
    d,_ = read_digit(img, cursor, unit_width, Lcode, Gcode, Rcode, "RIGHT", SPACE, BAR)
    digits += str(d)

  firstDigit = typeEncoding.get(encoding)
  print(str(firstDigit)+digits)

#imagen = 'barcodes/20170223_151821.jpg'
imagen = 'barcode5.jpg'
#imagen = 'barcode.png' ##Imagen wena pa k funke
read_barcode(imagen)

"""
# load the image and convert it to grayscale
image = cv2.imread('../images/' + imagen)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
#Se le aplica una resta para resaltar el resultado del eje X(Codigo de barras) frente al eje Y(numeros del codigo)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (5, 5))
bilateralFilter = cv2.bilateralFilter(gradient,9,75,75)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
(_, thresh1) = cv2.threshold(bilateralFilter, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

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
box = np.int0(cv2.boxPoints(rect))
 
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)


cv2.imshow('X',gradX)
cv2.imshow('Y',gradY)
cv2.imshow('',closed)
cv2.imshow('Blurred',blurred)
cv2.imshow('Threshold',thresh)
cv2.imshow('Bilateral',bilateralFilter)
cv2.imshow('Threshold 1',thresh1)
"""

cv2.waitKey(0)

cv2.destroyAllWindows()
