import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cvlib
import statistics as st
import argparse
import joblib

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
    #arr = np.full((row+(r*2), column+(r*2)), 255)
    #print(arr)
    arr[r:r+row, r:r+column] = image
    return arr
def view(img, title=None):
    k = 5
    fig,ax1 = plt.subplots(figsize=(k,k))
    if len(img.shape)==2:
        ax1.imshow(img, vmin=0, vmax=255, cmap='gray')
    else:
        ax1.imshow(img)
    if title:
        plt.title(title)   
    plt.axis('off')
    plt.show()

def contornos_hierarchy(imgray, b):
    """_summary_
        generamos los contornos y obtenemos la herencia de cada uno
    Args:
        imgray (image): image en escala gris
        b (int): valor de blur a aplicar

    Returns:
        _type_: retorna los contornos y las herencias de los mismos
    """
    b = b
    #blur 
    img_blur = cv.blur(imgray,(b,b))
    # adaptive Th
    imgbin = cv.adaptiveThreshold(img_blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,5)

    # compute contours
    mode = cv.RETR_TREE # contour retrieval mode
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] # contour approximation method 
    contours, hierarchy = cv.findContours(imgbin, mode, method[1])
    return contours, hierarchy

def get_placa(contours, h_img, w_img):
    """_summary_
        buscamos a lo largo de la imagen, si encontramos un contornos que cumpla con las caracterísitcas de una placa
        y luego la recortamos y regresamos la imagen
    Args:
        contours (_type_): contornos
        h_img (_type_): altura de la image
        w_img (_type_): ancho de la image

    Returns:
        _type_: retornamos el objeto que cuenta con el contorno de la placa
    """
    # detecto si tengo placa
    contorno = []
    for c in contours:
        # obtner extent
        area = cv.contourArea(c)
        x,y,w,h = cv.boundingRect(c)
        rect_area = w*h
        extent = float(area)/rect_area
        if (extent > 0.6): #pongo el mínimo del área
            if (w > (h*1.35) or w >= (h*1.2)) and ((w > 100 and w < 600) and (h > 50 and h < 330)):
                #print(f"el h: {h}")
                #print(f"el w: {w}")
                contorno.append(c)

    contorno_placa = []
    count = 0
    centro = [h_img/2, w_img/2]
    #print(centro)
    for c in contorno:
        if (count == 0):
            contorno_placa.append(c)
            count = 1
        else:
            x,y,w,h = cv.boundingRect(contorno_placa[0])
            x1,y1,w1,h1 = cv.boundingRect(c)
            xr1 = centro[1] - x
            xr2 = centro[1] - x1
            if (xr1 < xr2):
                pass
            else:
                contorno_placa.pop()
                contorno_placa.append(c)
    return contorno_placa
def imgnorm(img):
    """Nomalize an image
    Args:
        img (numpy array): Source image
    Returns:
        normalized (numpy array): Nomalized image
    """
    vmin, vmax = img.min(), img.max()
    normalized_values = []
    delta = vmax-vmin

    for p in img.ravel():
        normalized_values.append(255*(p-vmin)/delta)

    normalized  = np.array(normalized_values).astype(np.uint8).reshape(img.shape[0],-1)
    return normalized
def detect(contornos, indexes,placa):
    """_summary_
    con los contornos de las letras/numeros primero debemos ordenarlos, para ello vemos que se respede un patron
    primero van aquellos que esten en la línea más alta y que estén más a la izquierda. y luego vamos a la siguiente linea
    luego de tenerlos ordenamos, a cada letra la binarizamos de forma que le facilitemos al modelo su entendimiento. 
    Luego, el modelo nos retorna la letra/numero que considera y lo vamos guardando
    Args:
        contornos (_type_): contornos de las letras/numeros
        indexes (_type_): orden que vamos a leer los contornos
        placa (_type_): imagen de la placa a detectar para hacer los recortes

    Returns:
        _type_: retorna el array con el texto de la placa.
    """
    r = placa.copy()
    modelo = joblib.load("modelo_fotos.pkl")
    texto = []
    contornos_pos = []
    ys = []
    for cnt in range(len(contornos)):
            x,y,w,h = cv.boundingRect(contornos[cnt])
            contornos_pos.append([x,y, cnt])
            ys.append(y)
    miny = min(ys)
    maxy = max(ys)
    ys = []
    for ys_c in contornos_pos:
        if (ys_c[1] < (miny+5)):
            pos = contornos_pos.index(ys_c)
            contornos_pos[pos][1] = miny
            ys.append(miny)
        else:
            ys.append(maxy)
    contornos_pos = sorted(contornos_pos , key=lambda k: [k[1], k[0]])
    indexes = []
    for i in contornos_pos:
        indexes.append(i[2])
    for c in indexes:
        x,y,w,h = cv.boundingRect(contornos[c])
        r = r[y:y+h, x:x+w]
        r = cv.cvtColor(r, cv.COLOR_BGR2GRAY)
        r = imgnorm(r)
        r = cv.adaptiveThreshold(r, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,5)
        r = imgpad(r, 5)
        r = cv.resize(r, dsize=(81, 106), interpolation=cv.INTER_LANCZOS4)
        r = r.flatten()
        arr = []
        arr.append(r)
        respuesta = modelo.predict(arr)
        texto.append(respuesta[0])
        r = placa.copy()
    print(f"placa: {texto}")
    return texto
def main():
    image_path = ''
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--p", help="image path")

    args = argParser.parse_args()
    image_path = args.p
    filename = image_path
    im = cv.imread(filename)
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    h_img = imgray.shape[0]
    w_img = imgray.shape[1]
    ## obtener contrornos
    contours, hierarchy = contornos_hierarchy(imgray, 5)
    ## obtener placa
    contorno_placa = get_placa(contours, h_img, w_img)
    
    # valido si hay placa
    if (len(contorno_placa) != 0): # si tengo placa, extraigo
        # pongo los contornos con color
        colores = {}
        indexes = []
        thickness = 1
        for cnt in range(len(contorno_placa)):
            colores.update({cnt:(0,255,0)})
            indexes.append(cnt)
        placa = im.copy()
        x,y,w,h = cv.boundingRect(contorno_placa[0])
        placa = placa[y:y+h, x:x+w]
        imgray = cv.cvtColor(placa, cv.COLOR_BGR2GRAY)
        h_img = imgray.shape[0]
        w_img = imgray.shape[1]
        centro = [int(h_img/2), int(w_img/2)]
        # floddfill
        im_floodfill = imgray.copy()
        D = 65
        cv.floodFill(im_floodfill, None, (centro[1] ,centro[0]), 255,  loDiff=D, upDiff=D, flags=cv.FLOODFILL_FIXED_RANGE)
        imgbin = cv.adaptiveThreshold(im_floodfill, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,5)

        # compute contours
        mode = cv.RETR_TREE # contour retrieval mode
        method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] # contour approximation method 
        contours, hierarchy = cv.findContours(imgbin, mode, method[1])

        contorno = []
        var = -1
        count = 0
        #print(len(contours))
        while (len(contorno) < 3): 
            if count == 6:
                break
            for c in range(len(contours)):   
                # limitar el contorno, para tener un área más ajustada. 
                percent =0.052
                epsilon = percent*cv.arcLength(contours[c],True)
                approx = cv.approxPolyDP(contours[c],epsilon,True)
                if (cv.contourArea(approx)>300): #pongo el mínimo del área
                    x,y,w,h = cv.boundingRect(contours[c])
                    if (hierarchy[0][c][-1] == var):
                        if (w > (h*1.35) or w >= (h*1.2)):
                            pass
                        else:
                            contorno.append(contours[c])
            var = var + 1
            count = count + 1
        indexes = []
        for cnt in range(len(contorno)):
            indexes.append(cnt)
        texto = detect(contornos=contorno, indexes=indexes,placa=placa.copy())
        r = im.copy()
        h_img = r.shape[0]
        w_img = r.shape[1]
        for c in range(1):
            x,y,w,h = cv.boundingRect(contorno_placa[c])
            r = cv.rectangle(r,(x,y),(x+w,y+h),colores[c],2)
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (x, y-10)
            fontScale = 1
            color = (0, 255, 0)
            thickness = 3
            r = cv.putText(r, ' '.join(texto), org, font, 
                            fontScale, color, thickness, cv.LINE_AA)
        view(r, "Detectando placa")

    else: # si no detecto placa, de una voy a las letras
        contorno = []
        for c in range(len(contours)):   
        # limitar el contorno, para tener un área más ajustada. 
            percent =0.052
            epsilon = percent*cv.arcLength(contours[c],True)
            approx = cv.approxPolyDP(contours[c],epsilon,True)
            if (cv.contourArea(approx)>300): #pongo el mínimo del área
                x,y,w,h = cv.boundingRect(contours[c])
                if (hierarchy[0][c][-1] == -1 and (y > 10 and y < (int(h_img/2)) )):
                    if (w > (h*1.35) or w >= (h*1.2)):
                        #print("PASE POR AQUI")
                        pass
                    else:
                        contorno.append(contours[c])

        indexes = []
        for cnt in range(len(contorno)):
            indexes.append(cnt)
        texto = detect(contornos=contorno, indexes=indexes,placa=im.copy())
        r = im.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        x,y = 0,0
        y = int(h_img/3)
        x = int(w_img/3)
        org = (x, y)
        fontScale = 1
        color = (0, 255, 0)
        thickness = 3
        r = cv.putText(r, ' '.join(texto), org, font, 
                        fontScale, color, thickness, cv.LINE_AA)
        view(r, "SIN detectar placa")

if __name__ == "__main__":
    main()