# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:59:12 2017

@author: JM
"""
import cv2
import numpy as np
class barcode:
    def __init__(self,filename):
        #Imagen con el barcode recortado
        self.img = self.__barcodeProcessing(filename)
        #Definimos tres cursores como una lista con los ejes [X,Y]
        self.cursorS = [0,int(len(self.img) / 4)]
        self.cursorM = [0,int(len(self.img) / 2)]
        self.cursorI = [0,int(3*len(self.img) / 4)]
        #Se definen las tres regiones que se encuentran dentro del codigo de barras
        """
        L y G son diccionarios que se utilizan para el codigo de barras de la parte izquierda,
        en función de como se encuentren distribuidos estos en el codigo, el digito de seguridad
        viene definido a traves del tipo de codificacion.
        """
        self.Lcode = {"0001101":0,"0011001":1,"0010011":2,"0111101":3,"0100011":4,
                     "0110001":5,"0101111":6,"0111011":7,"0110111":8,"0001011":9}
          
        self.Gcode = {"0100111":0,"0110011":1,"0011011":2,"0100001":3,"0011101":4,
             "0111001":5,"0000101":6,"0010001":7,"0001001":8,"0010111":9}          
             
        self.Rcode = {"1110010":0,"1100110":1,"1101100":2,"1000010":3,"1011100":4,
             "1001110":5,"1010000":6,"1000100":7,"1001000":8,"1110100":9}
             
        self.typeEncoding = {"LLLLLL":0,"LLGLGG":1,"LLGGLG":2,"LLGGGL":3,"LGLLGG":4,
             "LGGLLG":5,"LGGGLL":6,"LGLGLG":7,"LGLGGL":8,"LGGLGL":9}
        #Definimos los espacios en negro y las barras de blanco
        self.SPACE = 0
        self.BAR = 255
    #Metodo para el procesamiento del barcode para su posterior lectura de barras
    def __barcodeProcessing(self,filename):
        img = cv2.imread('../images/' + filename, 0)
        img = cv2.bitwise_not(img)
        _,img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        #kernel = np.ones((1,1),np.uint8) / 9
        #img = cv2.erode(img,kernel,iterations = 1)
        #ret, img = cv2.threshold(img,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #kernel = np.ones((3,3),np.uint8)
        #img = cv2.dilate(img,kernel,iterations = 1)
        
        
        return img
    #Metodo para mostrar el barcode por pantalla
    def show(self):
        self.img[int(self.cursorS[1])]= 255
        self.img[self.cursorM[1]] = 255
        self.img[self.cursorI[1]] = 255
        cv2.imshow("imagen", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    """
    Alinea el cursor para que se situe en el primer bit de la zona izquierda o de la derecha,
    esto viene definido de la siguiente forma: 
        zona izquierda => Empieza siempre con bit en blanco y acaba en bit negro
        zona derecha => Empieza siempre con bit en negro y acaba en bit blanco
    """
    def __align_boundary(self,cursor, begin, end):
        if (self.img[cursor[1]][cursor[0]] == end):
            while (self.img[cursor[1]][cursor[0]] == end):
                cursor[0] += 1
        else:
            while (self.img[cursor[1]][cursor[0]-1] == begin):
                cursor[0] -= 1

    """
    Metodo que lee los 7 bits de cada digito
    Devuelve el digito asociado y el tipo de codificacion que tiene
    """
    def __read_digit(self, cursor, unit_width, position):
      pattern = [0, 0, 0, 0, 0, 0, 0] #Define los 7 bits del digito
      #Calcula el patron comprobando que se encuentra dentro de la anchura minima de las barras
      for i in range(len(pattern)):
        for j in range(unit_width):
          if (self.img[cursor[1]][cursor[0]] == self.BAR):
            pattern[i] += 1
          cursor[0] += 1
        #Permite que si estamos dentro de una barra y uno de esos pixeles tiene un valor diferente al resto
        #coloca el cursor un pixel atras ya que ese seria otra barra diferente.
        if (pattern[i] == 1 and self.img[cursor[1]][cursor[0]] == self.BAR 
            or pattern[i] == unit_width-1 and self.img[cursor[1]][cursor[0]] == self.SPACE):
          cursor[0] -= 1
     

      #Fija un umbral a partir del que consideramos que la lectura de pixeles hecha se corresponde con una barra negra
      threshold = unit_width / 2
      v = ""
      for i in range(len(pattern)):
        if pattern[i] >= threshold:
            v += str(1)
        else:
            v += str(0)
      
      encoding=""
      # En funcion de la zona en la que se encuentra, traduce los bits.
      if (position == "LEFT"):
        digit = self.Lcode.get(v)
        encoding = "L"
        if digit is None:
           digit = self.Gcode.get(v)
           encoding = "G"
        self.__align_boundary(cursor, self.SPACE, self.BAR)
      else:
        digit = self.Rcode.get(v)
        self.__align_boundary(cursor, self.BAR, self.SPACE) 
      #print("El codigo vale", v, "Digito", digit, "Pattern", pattern)

      return digit,encoding
  
    #Coloca el cursor en la primera zona de control
    def __skip_quiet_zone(self,cursor):
        while (self.img[cursor[1]][cursor[0]] == self.SPACE):
            cursor[0]+=1
  
    #Coloca el cursor tras la zona de control y calcula el tamaño minimo de las barras 
    def __read_lguard(self, cursor):
        widths = [ 0, 0, 0 ]
        pattern = [ self.BAR, self.SPACE, self.BAR ]
        for i in range(len(widths)):
            while (self.img[cursor[1]][cursor[0]] == pattern[i]):
                cursor[0]+=1
                widths[i]+=1
        return widths[0];
    
    #Salta los valores centrales
    def __skip_mguard(self, cursor):
        pattern = [ self.SPACE, self.BAR, self.SPACE, self.BAR, self.SPACE ]
        for i in range(len(pattern)):
            while (self.img[cursor[1]][cursor[0]] == pattern[i]):
                cursor[0]+=1


    def read_barcode(self):
        stopS,stopM,stopI = False,False,False
        #Nos saltamos la primera zona de seguridad
        self.__skip_quiet_zone(self.cursorS)
        self.__skip_quiet_zone(self.cursorM)
        self.__skip_quiet_zone(self.cursorI)
        #Calibramos el valor para que se coloque en la primera barra del codigo izquierdo y obtenemos la anchura de las barras
        unit_widthS = self.__read_lguard(self.cursorS)
        unit_widthM = self.__read_lguard(self.cursorM)
        unit_widthI = self.__read_lguard(self.cursorI)
 
        digits = ""
        encoding = ""
        """
        print("Valor eje X", self.cursor[0])
        for i in range(len(self.img)):        
            self.img[i][self.cursor[0]] = 255
        """
        #Se calculan los digitos y la codificacion de la parte izquierda
        for i in range(6):
            try:
                if stopS == False:
                    d1,e1 = self.__read_digit(self.cursorS, unit_widthS, "LEFT")
                if stopM == False:
                   d2,e2 = self.__read_digit(self.cursorM, unit_widthM, "LEFT")
                if stopI == False:
                   d3,e3 = self.__read_digit(self.cursorI, unit_widthI, "LEFT")
            except IndexError:
                if self.cursorS[0] >= len(self.img[0])-1:
                    stopS = True
                if self.cursorM[0] >= len(self.img[0])-1:
                    stopM = True
                if self.cursorI[0] >= len(self.img[0])-1:
                    stopI = True         
                      
            if (d1 == d2 or d1 == d3) and d1 is not None:
                d = d1
                e = e1
            elif (d2 == d1 or d2 == d3) and d2 is not None:
                d = d2
                e = e2
            elif (d3 == d1 or d3 == d2) and d3 is not None:
                d = d3
                e = e3
            elif d1 is not None:
                d = d1
                e = e1
            elif d2 is not None:
                d = d2
                e = e2
            elif d3 is not None:
                d = d3
                e = e3
            
            digits += str(d)
            encoding += e
        """
        print("Valor eje X", self.cursor[0])
        for i in range(len(self.img)):        
            self.img[i][self.cursor[0]] = 255
        """
        #Saltamos las barras de seguridad del medio
        self.__skip_mguard(self.cursorS)
        self.__skip_mguard(self.cursorM)
        self.__skip_mguard(self.cursorI)
 
        #Almacenamos los digitos asociados al valor del codigo derecho
        for i in range(6):
            try:
                if stopS == False:
                    d1,_ = self.__read_digit(self.cursorS, unit_widthS, "RIGHT")
                if stopM == False:
                   d2,_ = self.__read_digit(self.cursorM, unit_widthM, "RIGHT")
                if stopI == False:
                   d3,_ = self.__read_digit(self.cursorI, unit_widthI, "RIGHT")
            except IndexError:
                if self.cursorS[0] >= len(self.img[0])-1:
                    stopS = True
                if self.cursorM[0] >= len(self.img[0])-1:
                    stopM = True
                if self.cursorI[0] >= len(self.img[0])-1:
                    stopI = True 

            if (d1 == d2 or d1 == d3) and d1 is not None:
                d = d1
            elif (d2 == d1 or d2 == d3) and d2 is not None:
                d = d2
            elif (d3 == d1 or d3 == d2):
                d = d3
            elif d1 is not None:
                d = d1
            elif d2 is not None:
                d = d2
            elif d3 is not None:
                d = d3
            digits += str(d)
        firstDigit = self.typeEncoding.get(encoding)
        print(str(firstDigit)+digits)

 
imagen = 'barcode2.jpg'  
#imagen = 'barcode.png'     
barcode = barcode(imagen)
barcode.read_barcode()
barcode.show()