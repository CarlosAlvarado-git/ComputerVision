import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cvlib # mi librería
import random
import sys

labels_current = []


def imgpad(image, r):
    """_summary_
        La función empieza obteniendo el valor de columnas y filas de la imagen original. 
        Eso nos sirve, porque el padding es agregar un "marco" a la imagen, por lo que sabemos que
        r * 2 nos generara los espacios necesarios para el nuevo lenght de cada fila de la imagen. 
                                                                                    filas       columnas
        Por lo tanto, se crea un array con np.zero de numpy y sus dimensiones son: (row+(r*2), column+(r*2)
        Por ejemplo, si la imagen es de 8 * 8, tendremos una de 10 * 10, para tener 1 pixel de marco con el padding. 
        
        Para poner la imagen dentro de la matriz, le caemos encima a el espacio que equivale a donde estaría la imagen
        original. Eso nos trae justamente el espacio en donde corresponde nuestra imagen inicial. 
    Args:
        image (uint8): array de la imagen
        r (int): cantidad de pixeles que queremos nuestro padding. 
    """
    row,column = image.shape[0:2]
    arr = np.zeros((row+(r*2), column+(r*2)))
    arr[r:r+row, r:r+column] = image
    return arr

def find(data, i):
    """_summary_
        la función va de forma recursiva, hasta encontrar aquella posición que sea la del root del ID enviado
        la idea es encontrar justamente ese ID que está conectado con todos con los que el ID que se recibe está conectado,
        y determinar el más importante. 
    Args:
        data (int): array que cuenta con los labes (IDs) que se han encontrado por el first pass
        i (int): (ID) que se quiere encontrar su root. 

    Returns:
        _type_: _description_
    """
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

def union(data, i, j):
    """_summary_
        la función UNION lo que permite es que se encuentren los "root" de cada ID que se recibe y así encontrar
        cuál de los dos es el más importante, así asignamos ese en la posición que corresponde.

        todo se basa en posiciones, es decir, el ID 1 que manda el usuario, realmente es la posición 0 y así sucesivamente 

    Args:
        data (int): array que cuenta con los labes (IDs) que se han encontrado por el first pass
        i (int): (ID) menor del conflicto
        j (int): (ID) mayor del conflicto
    """
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pj] = pi

def connected_c(img):
    """_summary_
        La función realiza el connected component, de forma de Second Pass. 
        Esto porque realiza dos pasadas dentro de la matriz, para así identificar aquellas regiones que están 
        conectadas. 
        First pass: es la que almacena los labels_current(IDs) que se van econtrando y encuentra los conflictos. 
            se utiliza la función UNION al memento de econtrar un conflicto entre 2 IDs
        Seconda pass: ya teniendo los conflictos y las labels (IDs) se vuelve a recorrer la matriz y se utiliza
            la función FIND, que encuentra que ID root debe ser asignada por cada pixel. 
            El root es aquel ID más pequeño que hace referencia al segmento más grande. 
    Args:
        img (any): la imagen convertidad

    Returns:
        matriz/int: la matriz que cuenta con los IDs, con los segmentos más grandes. 
    """
    # first_pass
    global labels_current
    labels_current = []
    current_pixel = 1
    conflictos = []
    copy_image = img.copy()
    copy_image[copy_image>0] = 1
    # hacemos un pad, para poder determinar aquellos labes de las esquinas. 
    copy_image = imgpad(copy_image, 1)
    row, column = copy_image.shape[0:2]
    for r in range(row):
        for c in range(column):
                if (copy_image[r,c] != 0):
                    vecinos = [copy_image[r,c-1], copy_image[r-1,c]]
                    #print(vecinos)
                    if (vecinos[0] == 0 and vecinos[1] == 0):
                        copy_image[r,c] = int(current_pixel)
                        labels_current.append(current_pixel-1)
                        current_pixel += 1
                        #print(f"NO hay izquierda ni arriba, en pos: {r,c}")
                    else: 
                        if (vecinos[0] != 0 and vecinos[1] != 0):
                            if (vecinos[0] == vecinos[1]):
                                copy_image[r,c] = int(vecinos[0])
                            else:
                                copy_image[r,c] = int(min(vecinos))
                                if [int(min(vecinos)), int(max(vecinos))] in conflictos:
                                    pass
                                else:
                                    conflictos.append((int(min(vecinos))-1, int(max(vecinos))-1))
                                # crear relacion
                        elif(vecinos[0] == 0):
                             copy_image[r,c] = int(vecinos[1])
                        else:
                             copy_image[r,c] = int(vecinos[0])
    conflictos = list(set(conflictos))
    for i, j in conflictos:
        union(labels_current, i, j)
    copy_image = copy_image.astype(int)
    # two_pass
    for r in range(row):
        for c in range(column):
                if (copy_image[r,c] != 0):
                    copy_image[r,c] = (find(labels_current, (copy_image[r,c]-1)))+1
                    
    #print(labels_current)
    #labels_current = [i + value for i in labels_current]
    return copy_image

def random_color():
    """_summary_
        función que genera un color dentro del rango y lo retorna. 
    Returns:
        _type_: retorna una tupla para generar un color, de forma RGB
    """
    color = range(32,256,32)
    return tuple(random.choice(color) for _ in range(3))

def labelview(labels):
    """_summary_
        labelview es una función que utiliza la función random_color para genera un color nuevo para cada ID
        que se a identificado en la imagen. Luego, se crea una matriz de 0, pero que permite el ingreso de tuplas.
        Eso para que al momeno de encontrar un ID distinto a 0, se le asigne el color que se le asigno. 
    Args:
        labels (int): matriz de int, que cuenta con los IDs en cada "pixel"
    """
    colores = {}
    row, column = labels.shape[0:2]
    img_color =  np.zeros(labels.shape, dtype=np.uint8)
    img_color = cv.cvtColor(img_color, cv.COLOR_GRAY2RGB)
    lista_is = list(set(labels_current))
    for i in [i + 1 for i in lista_is]:
        colores[i] = random_color()
    for r in range(row):
        for c in range(column):
            if (labels[r,c] != 0):
                img_color[r,c] = colores[labels[r,c]]
    #cvlib.imgview(img_color)
    return img_color

def main():
    global labels_current
    imagen_nombre = sys.argv[1]
    imagen_guardar = sys.argv[2]
    print(f"el imagen_nombre {imagen_nombre}, el imagen_guardar {imagen_guardar}")
    img = cv.imread(imagen_nombre, cv.IMREAD_GRAYSCALE)
    r,c = img.shape[0:2]
    #print('Rows {0}\nColumns {1}\nPixels {2:,}'.format(r,c,r*c))

    # necesitamos eliminar el ruido que llegamos a producir con el adaptive threshold. 
    im_floodfill = img.copy()
    D = 30
    cv.floodFill(im_floodfill, None, (10 ,10), 255,  loDiff=D, upDiff=D, flags=cv.FLOODFILL_FIXED_RANGE)
    #cvlib.imgview(im_floodfill)

    img_result = cv.adaptiveThreshold(im_floodfill, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,41,2)
    mode = cv.RETR_TREE # contour retrieval mode
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] # contour approximation method 
    contours, hierarchy = cv.findContours(img_result, mode, method[1])

    # "pintamos" los contornos de la imagen, para así volver las líneas más delgadas y poder hacer una mejor 
    # diferenciación de las lineas de la huella. 
    index = -1
    color = (0,255,0) #(r,g,b)
    thickness = 1
    imgcont = cv.drawContours(img_result.copy(), contours, index, color, thickness)
    #cvlib.imgview(imgcont)
    img_result = imgcont

    # uso de imgpad, solo es ejemplo:
    img2 = imgpad(img_result, 1)
    r,c = img.shape[0:2]
    r2,c2 = img2.shape[0:2]
    #cvlib.imgcmp(img_result, img2, title=['Orignal row: {0}, column: {1}'.format(r,c),'Padding row: {0}, column: {1}'.format(r2,c2)])
    
    #uso de connected_c
    img_ccl = connected_c(img_result)

    img_final = labelview(img_ccl)
    cvlib.imgview(img_final, filename=imagen_guardar)

if __name__ == "__main__":
    main()