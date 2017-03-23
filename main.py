# lista de proveedores
import numpy as np

def inicia():
		
	prov1 =[2, 100, 249, 90, 100]
	prov2 =[13, 99.97, 643, 80, 100]
	prov3 =[3, 100, 714, 90, 100]
	prov4 =[3, 100, 1809, 90, 100]
	prov5 =[24, 99.83, 238, 90, 100]
	prov6 =[28, 96.59, 241, 90, 100]
	prov7 =[1,100,1404,85,100]
	prov8 =[24,100,984,97,100]
	prov9 =[11,99.91,641,90,100]
	prov10 =[53,97.54,588,100,100]
	prov11 =[10,99.95,241,95,100]
	prov12 =[7,99.85,567,98,100]
	prov13 =[19,99.97,567,90,100]
	prov14 =[12,91.89,967,90,100]
	prov15 =[33,99.99,635,95,80]
	prov16 =[2,100,795,95,100]
	prov17 =[34,99.99,689,95,80]
	prov18 =[9,99.36,913,85,100]
	
	prov =np.array([prov1,prov2,prov3,prov4,prov5,prov6,prov7,prov8,prov9,prov10,prov11,prov12,prov13,prov14,prov15,prov16,prov17,prov18])
	return prov

# normaliza las características
def normalizaLista(lista):
			
	normalized = lista.copy()
	for i in range(len(lista)):						
		for j in range(len(lista[i])):					
			valMax = max (lista[:,j])	
			valMin = min (lista[:,j])		
			val = lista[i][j] 	
			normalized[i][j] = (val-valMin)/(valMax-valMin)			

	#print(normalized,"\n")	
	
	return(normalized)
		#for j in range(len(lista)):

#este metodo establece la proporcion inversa de los valores selecionados
# en este caso los valores que hay que invertir son la distancia reciproca y el indice de precios (columnas 3 y 5)			
def invierte_proporcion(lista,listaValores):
	
	for valor in listaValores:
		lista[:,valor]=abs(lista[:,valor]-1)
	
	return (lista)

def peso_proveedor(listaNormalizada):
	pesos = [[] for _ in listaNormalizada]
		
	for i in range(len(listaNormalizada)):
				
		for j in range(len(listaNormalizada[i])):
			suma = 0			
			for z in range(j+1):
				suma +=listaNormalizada[i][z]
			suma = suma/(j+1)
			pesos[i].append(suma)	
	listaProveedores = [max(i) for i in pesos]
	print(pesos,"\n")
	print(listaProveedores)



listaOri = inicia()
lista = normalizaLista(inicia())
lista = invierte_proporcion(lista, [2,4])
peso_proveedor(lista)







